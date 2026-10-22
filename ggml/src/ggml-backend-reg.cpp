#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-backend-dl.h"
#شامل "ggml-impl.h"
#شامل <algorithm>
#شامل <cstring>
#شامل <filesystem>
#شامل <memory>
#شامل <string>
#شامل <type_traits>
#شامل <vector>
#شامل <cctype>

#اگر_تعریف_شدہ _WIN32
#    define WIN32_LEAN_AND_اوسط
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#وگرنہ_اگر defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#ورنہ
#    include <dlfcn.h>
#    include <unistd.h>
#اگر_ختم

// Backend registry
#اگر_تعریف_شدہ GGML_USE_CPU
#شامل "ggml-cpu.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CUDA
#شامل "ggml-cuda.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_METAL
#شامل "ggml-metal.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_SYCL
#شامل "ggml-sycl.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_VULKAN
#شامل "ggml-vulkan.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_WEBGPU
#شامل "ggml-webgpu.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_ZDNN
#شامل "ggml-zdnn.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_OPENCL
#شامل "ggml-opencl.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_HEXAGON
#شامل "ggml-hexagon.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_BLAS
#شامل "ggml-blas.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_RPC
#شامل "ggml-rpc.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_VIRTGPU_FRONTEND
#شامل "ggml-virtgpu.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_CANN
#شامل "ggml-cann.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_ZENDNN
#شامل "ggml-zendnn.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_OPENVINO
#شامل "ggml-openvino.h"
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_ET
#شامل "ggml-et.h"
#اگر_ختم

نامخلا fs = std::filesystem;

ساکن std::string راستہ_str(ثابت fs::path & path) {
    کوشش {
#اگر defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ثابت std::u8string u8str = path.u8string();
        واپس std::string(reinterpret_cast<ثابت حرف *>(u8str.ڈیٹا()), u8str.size());
#ورنہ
        // C++17: u8string() returns std::string
        واپس path.u8string();
#اگر_ختم
    } پکڑو (...) {
        واپس std::string();
    }
}

ساخت ggml_backend_reg_اندراج {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

ساخت ggml_backend_registry {
    std::vector<ggml_backend_reg_اندراج> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#اگر_تعریف_شدہ GGML_USE_CUDA
        رجسٹر_backend(ggml_backend_cuda_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_METAL
        رجسٹر_backend(ggml_backend_metal_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_SYCL
        رجسٹر_backend(ggml_backend_sycl_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_VULKAN
    // Add runtime disable check
    اگر (getenv("GGML_DISABLE_VULKAN") == خالی_اشاریہ) {
        رجسٹر_backend(ggml_backend_vk_reg());
    } ورنہ {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_WEBGPU
        رجسٹر_backend(ggml_backend_webgpu_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_ZDNN
        رجسٹر_backend(ggml_backend_zdnn_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_VIRTGPU_FRONTEND
        رجسٹر_backend(ggml_backend_virtgpu_reg());
#اگر_ختم

#اگر_تعریف_شدہ GGML_USE_OPENCL
        رجسٹر_backend(ggml_backend_opencl_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_ZENDNN
        رجسٹر_backend(ggml_backend_zendnn_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_HEXAGON
        رجسٹر_backend(ggml_backend_hexagon_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_CANN
        رجسٹر_backend(ggml_backend_cann_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_BLAS
        رجسٹر_backend(ggml_backend_blas_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_RPC
        رجسٹر_backend(ggml_backend_rpc_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_OPENVINO
        رجسٹر_backend(ggml_backend_openvino_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_ET
        رجسٹر_backend(ggml_backend_et_reg());
#اگر_ختم
#اگر_تعریف_شدہ GGML_USE_CPU
        رجسٹر_backend(ggml_backend_cpu_reg());
#اگر_ختم
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        برائے (خودکار & اندراج_2 : backends) {
            اگر (اندراج_2.handle) {
                اندراج_2.handle.release(); // NOLINT
            }
        }
    }

    خالی رجسٹر_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = خالی_اشاریہ) {
        اگر (!reg) {
            واپس;
        }

        برائے (خودکار & اندراج_2 : backends) {
            اگر (اندراج_2.reg == reg) {
                واپس;
            }
        }

#اگر_غیر_تعریف_شدہ NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_گنتی(reg));
#اگر_ختم
        backends.push_back({ reg, std::move(handle) });
        برائے (size_t i = 0; i < ggml_backend_reg_dev_گنتی(reg); i++) {
            رجسٹر_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    خالی رجسٹر_device(ggml_backend_dev_t device) {
        برائے (خودکار & dev : devices) {
            اگر (dev == device) {
                واپس;
            }
        }

#اگر_غیر_تعریف_شدہ NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#اگر_ختم
        devices.push_back(device);
    }

    ggml_backend_reg_t بھرو_backend(ثابت fs::path & path, منطقی silent) {
        dl_handle_ptr handle { dl_بھرو_library(path) };
        اگر (!handle) {
            اگر (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, راستہ_str(path).c_str(), dl_error());
            }
            واپس خالی_اشاریہ;
        }

        خودکار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        اگر (score_fn && score_fn() == 0) {
            اگر (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, راستہ_str(path).c_str());
            }
            واپس خالی_اشاریہ;
        }

        خودکار backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        اگر (!backend_init_fn) {
            اگر (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, راستہ_str(path).c_str());
            }
            واپس خالی_اشاریہ;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        اگر (!reg || reg->api_نسخہ != GGML_BACKEND_API_نسخہ) {
            اگر (!silent) {
                اگر (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, راستہ_str(path).c_str());
                } ورنہ {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, راستہ_str(path).c_str(), reg->api_نسخہ, GGML_BACKEND_API_نسخہ);
                }
            }
            واپس خالی_اشاریہ;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), راستہ_str(path).c_str());

        رجسٹر_backend(reg, std::move(handle));

        واپس reg;
    }

    خالی unload_backend(ggml_backend_reg_t reg, منطقی silent) {
        خودکار it = std::find_if(backends.begin(), backends.end(),
                               [reg](ثابت ggml_backend_reg_اندراج & اندراج_2) { واپس اندراج_2.reg == reg; });

        اگر (it == backends.end()) {
            اگر (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            واپس;
        }

        اگر (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { واپس ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ساکن ggml_backend_registry & get_reg() {
    ساکن ggml_backend_registry reg;
    واپس reg;
}

// Internal API
خالی ggml_backend_رجسٹر(ggml_backend_reg_t reg) {
    get_reg().رجسٹر_backend(reg);
}

خالی ggml_backend_device_رجسٹر(ggml_backend_dev_t device) {
    get_reg().رجسٹر_device(device);
}

// Backend (reg) enumeration
ساکن منطقی striequals(ثابت حرف * a, ثابت حرف * b) {
    برائے (; *a && *b; a++, b++) {
        اگر (std::tolower(*a) != std::tolower(*b)) {
            واپس جھوٹ;
        }
    }
    واپس *a == *b;
}

size_t ggml_backend_reg_گنتی() {
    واپس get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t اشاریہ) {
    GGML_ASSERT(اشاریہ < ggml_backend_reg_گنتی());
    واپس get_reg().backends[اشاریہ].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ثابت حرف * name) {
    برائے (size_t i = 0; i < ggml_backend_reg_گنتی(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        اگر (striequals(ggml_backend_reg_name(reg), name)) {
            واپس reg;
        }
    }
    واپس خالی_اشاریہ;
}

// Device enumeration
size_t ggml_backend_dev_گنتی() {
    واپس get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t اشاریہ) {
    GGML_ASSERT(اشاریہ < ggml_backend_dev_گنتی());
    واپس get_reg().devices[اشاریہ];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ثابت حرف * name) {
    برائے (size_t i = 0; i < ggml_backend_dev_گنتی(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        اگر (striequals(ggml_backend_dev_name(dev), name)) {
            واپس dev;
        }
    }
    واپس خالی_اشاریہ;
}

ggml_backend_dev_t ggml_backend_dev_by_type(شمار ggml_backend_dev_type type) {
    برائے (size_t i = 0; i < ggml_backend_dev_گنتی(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        اگر (ggml_backend_dev_type(dev) == type) {
            واپس dev;
        }
    }
    واپس خالی_اشاریہ;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ثابت حرف * name, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    اگر (!dev) {
        واپس خالی_اشاریہ;
    }
    واپس ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(شمار ggml_backend_dev_type type, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    اگر (!dev) {
        واپس خالی_اشاریہ;
    }
    واپس ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_بہترین(خالی) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    اگر (!dev) {
        واپس خالی_اشاریہ;
    }
    واپس ggml_backend_dev_init(dev, خالی_اشاریہ);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_بھرو(ثابت حرف * path) {
    واپس get_reg().بھرو_backend(path, جھوٹ);
}

خالی ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, سچ);
}

ساکن fs::path get_executable_راستہ() {
#اگر defined(__APPLE__)
    // get executable path
    std::vector<حرف> path;
    uint32_t size;
    جب (سچ) {
        size = path.size();
        اگر (NSGet_Executable_راستہ(path.ڈیٹا(), &size) == 0) {
            توڑو;
        }
        path.resize(size);
    }
    std::string بنیاد_راستہ(path.ڈیٹا(), size);
    // remove executable name
    خودکار آخری_slash = بنیاد_راستہ.find_last_of('/');
    اگر (آخری_slash != std::string::npos) {
        بنیاد_راستہ = بنیاد_راستہ.substr(0, آخری_slash);
    }
    واپس بنیاد_راستہ + "/";
#وگرنہ_اگر defined(__linux__) || defined(__FreeBSD__)
    std::string بنیاد_راستہ = ".";
    std::vector<حرف> path(1024);
    جب (سچ) {
        // get executable path
#    اگر defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.ڈیٹا(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.ڈیٹا(), path.size());
#    endif
        اگر (len == -1) {
            توڑو;
        }
        اگر (len < (ssize_t) path.size()) {
            بنیاد_راستہ = std::string(path.ڈیٹا(), len);
            // remove executable name
            خودکار آخری_slash = بنیاد_راستہ.find_last_of('/');
            اگر (آخری_slash != std::string::npos) {
                بنیاد_راستہ = بنیاد_راستہ.substr(0, آخری_slash);
            }
            توڑو;
        }
        path.resize(path.size() * 2);
    }

    واپس بنیاد_راستہ + "/";
#وگرنہ_اگر defined(_WIN32)
    std::vector<wchar_t> path(MAX_راستہ);
    DWORD len = GetModuleFileNameW(تہی, path.ڈیٹا(), path.size());
    اگر (len == 0) {
        واپس {};
    }
    std::wstring بنیاد_راستہ(path.ڈیٹا(), len);
    // remove executable name
    خودکار آخری_slash = بنیاد_راستہ.find_last_of('\\');
    اگر (آخری_slash != std::string::npos) {
        بنیاد_راستہ = بنیاد_راستہ.substr(0, آخری_slash);
    }
    واپس بنیاد_راستہ + L"\\";
#ورنہ
    واپس {};
#اگر_ختم
}

ساکن fs::path backend_filename_سابقہ() {
#اگر_تعریف_شدہ _WIN32
    واپس fs::u8path("ggml-");
#ورنہ
    واپس fs::u8path("libggml-");
#اگر_ختم
}

ساکن fs::path backend_filename_extension() {
#اگر_تعریف_شدہ _WIN32
    واپس fs::u8path(".dll");
#ورنہ
    واپس fs::u8path(".so");
#اگر_ختم
}

ساکن ggml_backend_reg_t ggml_backend_بھرو_بہترین(ثابت حرف * name, منطقی silent, ثابت حرف * user_تلاش_راستہ) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ثابت fs::path name_راستہ = fs::u8path(name);
    ثابت fs::path file_سابقہ = backend_filename_سابقہ().native() + name_راستہ.native() + fs::u8path("-").native();
    ثابت fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> تلاش_paths;
    اگر (user_تلاش_راستہ == خالی_اشاریہ) {
#اگر_تعریف_شدہ GGML_BACKEND_DIR
        تلاش_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#اگر_ختم
        // default search paths: executable directory, current directory
        تلاش_paths.push_back(get_executable_راستہ());
        std::error_code cwd_ec;
        ثابت fs::path cwd = fs::current_path(cwd_ec);
        اگر (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.پیغام().c_str());
        } ورنہ {
            تلاش_paths.push_back(cwd);
        }
    } ورنہ {
        تلاش_paths.push_back(fs::u8path(user_تلاش_راستہ));
    }

    صحیح بہترین_score = 0;
    fs::path بہترین_راستہ;
    std::error_code ec;

    برائے (ثابت خودکار & تلاش_راستہ : تلاش_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(تلاش_راستہ, dir_ec);

        برائے (ثابت fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ثابت خودکار & اندراج_2 = *dir_it;
            اگر (اندراج_2.is_regular_file(ec)) {
                خودکار filename = اندراج_2.path().filename();
                خودکار ext = اندراج_2.path().extension();
                اگر (filename.native().find(file_سابقہ) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_بھرو_library(اندراج_2) };
                    اگر (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, راستہ_str(اندراج_2.path()).c_str(), dl_error());
                    }
                    اگر (handle) {
                        خودکار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        اگر (score_fn) {
                            صحیح s = score_fn();
#اگر_غیر_تعریف_شدہ NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, راستہ_str(اندراج_2.path()).c_str(), s);
#اگر_ختم
                            اگر (s > بہترین_score) {
                                بہترین_score = s;
                                بہترین_راستہ = اندراج_2.path();
                            }
                        } ورنہ {
                            اگر (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, راستہ_str(اندراج_2.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        اگر (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, راستہ_str(تلاش_راستہ).c_str(), dir_ec.پیغام().c_str());
        }
    }

    اگر (بہترین_score == 0) {
        // try to load the base backend
        برائے (ثابت خودکار & تلاش_راستہ : تلاش_paths) {
            fs::path filename = backend_filename_سابقہ().native() + name_راستہ.native() + backend_filename_extension().native();
            fs::path path = تلاش_راستہ / filename;
            اگر (std::error_code ec; fs::exists(path, ec)) {
                واپس get_reg().بھرو_backend(path, silent);
            } ورنہ {
                اگر (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, راستہ_str(path).c_str(), ec.پیغام().c_str());
                }
            }
        }
        واپس خالی_اشاریہ;
    }

    واپس get_reg().بھرو_backend(بہترین_راستہ, silent);
}

خالی ggml_backend_بھرو_all() {
    ggml_backend_بھرو_all_from_راستہ(خالی_اشاریہ);
}

خالی ggml_backend_بھرو_all_from_راستہ(ثابت حرف * dir_راستہ) {
#اگر_تعریف_شدہ NDEBUG
    منطقی silent = سچ;
#ورنہ
    منطقی silent = جھوٹ;
#اگر_ختم

    ggml_backend_بھرو_بہترین("blas", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("zendnn", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("cann", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("cuda", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("hip", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("metal", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("rpc", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("sycl", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("vulkan", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("virtgpu", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("opencl", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("hexagon", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("musa", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("openvino", silent, dir_راستہ);
    ggml_backend_بھرو_بہترین("cpu", silent, dir_راستہ);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ثابت حرف * backend_راستہ = std::getenv("GGML_BACKEND_PATH");
    اگر (backend_راستہ) {
        ggml_backend_بھرو(backend_راستہ);
    }
}
