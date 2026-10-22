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

#جے_تعریف_شدہ _WIN32
#    define WIN32_LEAN_AND_اوسط
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
#جے_انت

// Backend registry
#جے_تعریف_شدہ GGML_USE_CPU
#ضمن "ggml-cpu.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_CUDA
#ضمن "ggml-cuda.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_METAL
#ضمن "ggml-metal.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_SYCL
#ضمن "ggml-sycl.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_VULKAN
#ضمن "ggml-vulkan.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_WEBGPU
#ضمن "ggml-webgpu.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_ZDNN
#ضمن "ggml-zdnn.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_OPENCL
#ضمن "ggml-opencl.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_HEXAGON
#ضمن "ggml-hexagon.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_BLAS
#ضمن "ggml-blas.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_RPC
#ضمن "ggml-rpc.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_VIRTGPU_FRONTEND
#ضمن "ggml-virtgpu.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_CANN
#ضمن "ggml-cann.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_ZENDNN
#ضمن "ggml-zendnn.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_OPENVINO
#ضمن "ggml-openvino.h"
#جے_انت

#جے_تعریف_شدہ GGML_USE_ET
#ضمن "ggml-et.h"
#جے_انت

نطاق fs = std::filesystem;

ساكن std::string رستہ_str(ثابت fs::path & path) {
    حاول {
#إذا defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ثابت std::u8string u8str = path.u8string();
        أرجع std::string(reinterpret_cast<ثابت حرف *>(u8str.انکڑے()), u8str.size());
#وإلا
        // C++17: u8string() returns std::string
        أرجع path.u8string();
#جے_انت
    } امسك (...) {
        أرجع std::string();
    }
}

بنية ggml_backend_reg_اندراج {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

بنية ggml_backend_registry {
    std::vector<ggml_backend_reg_اندراج> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#جے_تعریف_شدہ GGML_USE_CUDA
        رجسٹر_backend(ggml_backend_cuda_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_METAL
        رجسٹر_backend(ggml_backend_metal_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_SYCL
        رجسٹر_backend(ggml_backend_sycl_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_VULKAN
    // Add runtime disable check
    إذا (getenv("GGML_DISABLE_VULKAN") == خالی_سوچک) {
        رجسٹر_backend(ggml_backend_vk_reg());
    } وإلا {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#جے_انت
#جے_تعریف_شدہ GGML_USE_WEBGPU
        رجسٹر_backend(ggml_backend_webgpu_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_ZDNN
        رجسٹر_backend(ggml_backend_zdnn_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_VIRTGPU_FRONTEND
        رجسٹر_backend(ggml_backend_virtgpu_reg());
#جے_انت

#جے_تعریف_شدہ GGML_USE_OPENCL
        رجسٹر_backend(ggml_backend_opencl_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_ZENDNN
        رجسٹر_backend(ggml_backend_zendnn_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_HEXAGON
        رجسٹر_backend(ggml_backend_hexagon_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_CANN
        رجسٹر_backend(ggml_backend_cann_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_BLAS
        رجسٹر_backend(ggml_backend_blas_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_RPC
        رجسٹر_backend(ggml_backend_rpc_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_OPENVINO
        رجسٹر_backend(ggml_backend_openvino_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_ET
        رجسٹر_backend(ggml_backend_et_reg());
#جے_انت
#جے_تعریف_شدہ GGML_USE_CPU
        رجسٹر_backend(ggml_backend_cpu_reg());
#جے_انت
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        لكل (خودکار & اندراج : backends) {
            إذا (اندراج.handle) {
                اندراج.handle.release(); // NOLINT
            }
        }
    }

    فراغ رجسٹر_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = خالی_سوچک) {
        إذا (!reg) {
            أرجع;
        }

        لكل (خودکار & اندراج : backends) {
            إذا (اندراج.reg == reg) {
                أرجع;
            }
        }

#جے_بنا_تعریف NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_گنتی(reg));
#جے_انت
        backends.push_back({ reg, std::move(handle) });
        لكل (size_t i = 0; i < ggml_backend_reg_dev_گنتی(reg); i++) {
            رجسٹر_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    فراغ رجسٹر_device(ggml_backend_dev_t device) {
        لكل (خودکار & dev : devices) {
            إذا (dev == device) {
                أرجع;
            }
        }

#جے_بنا_تعریف NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#جے_انت
        devices.push_back(device);
    }

    ggml_backend_reg_t لدو_backend(ثابت fs::path & path, منطقي silent) {
        dl_handle_ptr handle { dl_لدو_library(path) };
        إذا (!handle) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, رستہ_str(path).c_str(), dl_error());
            }
            أرجع خالی_سوچک;
        }

        خودکار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        إذا (score_fn && score_fn() == 0) {
            إذا (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, رستہ_str(path).c_str());
            }
            أرجع خالی_سوچک;
        }

        خودکار backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        إذا (!backend_init_fn) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, رستہ_str(path).c_str());
            }
            أرجع خالی_سوچک;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        إذا (!reg || reg->api_نسخہ != GGML_BACKEND_API_نسخہ) {
            إذا (!silent) {
                إذا (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, رستہ_str(path).c_str());
                } وإلا {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, رستہ_str(path).c_str(), reg->api_نسخہ, GGML_BACKEND_API_نسخہ);
                }
            }
            أرجع خالی_سوچک;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), رستہ_str(path).c_str());

        رجسٹر_backend(reg, std::move(handle));

        أرجع reg;
    }

    فراغ unload_backend(ggml_backend_reg_t reg, منطقي silent) {
        خودکار it = std::find_if(backends.begin(), backends.end(),
                               [reg](ثابت ggml_backend_reg_اندراج & اندراج) { أرجع اندراج.reg == reg; });

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
فراغ ggml_backend_رجسٹر(ggml_backend_reg_t reg) {
    get_reg().رجسٹر_backend(reg);
}

فراغ ggml_backend_device_رجسٹر(ggml_backend_dev_t device) {
    get_reg().رجسٹر_device(device);
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

size_t ggml_backend_reg_گنتی() {
    أرجع get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t اشاریہ) {
    GGML_ASSERT(اشاریہ < ggml_backend_reg_گنتی());
    أرجع get_reg().backends[اشاریہ].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_reg_گنتی(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        إذا (striequals(ggml_backend_reg_name(reg), name)) {
            أرجع reg;
        }
    }
    أرجع خالی_سوچک;
}

// Device enumeration
size_t ggml_backend_dev_گنتی() {
    أرجع get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t اشاریہ) {
    GGML_ASSERT(اشاریہ < ggml_backend_dev_گنتی());
    أرجع get_reg().devices[اشاریہ];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_dev_گنتی(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (striequals(ggml_backend_dev_name(dev), name)) {
            أرجع dev;
        }
    }
    أرجع خالی_سوچک;
}

ggml_backend_dev_t ggml_backend_dev_by_type(تعداد ggml_backend_dev_type type) {
    لكل (size_t i = 0; i < ggml_backend_dev_گنتی(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (ggml_backend_dev_type(dev) == type) {
            أرجع dev;
        }
    }
    أرجع خالی_سوچک;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ثابت حرف * name, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    إذا (!dev) {
        أرجع خالی_سوچک;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(تعداد ggml_backend_dev_type type, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    إذا (!dev) {
        أرجع خالی_سوچک;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_بہترین(فراغ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    إذا (!dev) {
        أرجع خالی_سوچک;
    }
    أرجع ggml_backend_dev_init(dev, خالی_سوچک);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_لدو(ثابت حرف * path) {
    أرجع get_reg().لدو_backend(path, خطأ);
}

فراغ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, سچ);
}

ساكن fs::path get_executable_رستہ() {
#إذا defined(__APPLE__)
    // get executable path
    std::vector<حرف> path;
    uint32_t size;
    طالما (سچ) {
        size = path.size();
        إذا (NSGet_Executable_رستہ(path.انکڑے(), &size) == 0) {
            اكسر;
        }
        path.resize(size);
    }
    std::string ادھار_رستہ(path.انکڑے(), size);
    // remove executable name
    خودکار آخری_slash = ادھار_رستہ.find_last_of('/');
    إذا (آخری_slash != std::string::npos) {
        ادھار_رستہ = ادھار_رستہ.substr(0, آخری_slash);
    }
    أرجع ادھار_رستہ + "/";
#وإلاإذا defined(__linux__) || defined(__FreeBSD__)
    std::string ادھار_رستہ = ".";
    std::vector<حرف> path(1024);
    طالما (سچ) {
        // get executable path
#    إذا defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.انکڑے(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.انکڑے(), path.size());
#    endif
        إذا (len == -1) {
            اكسر;
        }
        إذا (len < (ssize_t) path.size()) {
            ادھار_رستہ = std::string(path.انکڑے(), len);
            // remove executable name
            خودکار آخری_slash = ادھار_رستہ.find_last_of('/');
            إذا (آخری_slash != std::string::npos) {
                ادھار_رستہ = ادھار_رستہ.substr(0, آخری_slash);
            }
            اكسر;
        }
        path.resize(path.size() * 2);
    }

    أرجع ادھار_رستہ + "/";
#وإلاإذا defined(_WIN32)
    std::vector<wchar_t> path(MAX_رستہ);
    DWORD len = GetModuleFileNameW(عدم, path.انکڑے(), path.size());
    إذا (len == 0) {
        أرجع {};
    }
    std::wstring ادھار_رستہ(path.انکڑے(), len);
    // remove executable name
    خودکار آخری_slash = ادھار_رستہ.find_last_of('\\');
    إذا (آخری_slash != std::string::npos) {
        ادھار_رستہ = ادھار_رستہ.substr(0, آخری_slash);
    }
    أرجع ادھار_رستہ + L"\\";
#وإلا
    أرجع {};
#جے_انت
}

ساكن fs::path backend_filename_اگیتر() {
#جے_تعریف_شدہ _WIN32
    أرجع fs::u8path("ggml-");
#وإلا
    أرجع fs::u8path("libggml-");
#جے_انت
}

ساكن fs::path backend_filename_extension() {
#جے_تعریف_شدہ _WIN32
    أرجع fs::u8path(".dll");
#وإلا
    أرجع fs::u8path(".so");
#جے_انت
}

ساكن ggml_backend_reg_t ggml_backend_لدو_بہترین(ثابت حرف * name, منطقي silent, ثابت حرف * user_تلاش_رستہ) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ثابت fs::path name_رستہ = fs::u8path(name);
    ثابت fs::path file_اگیتر = backend_filename_اگیتر().native() + name_رستہ.native() + fs::u8path("-").native();
    ثابت fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> تلاش_paths;
    إذا (user_تلاش_رستہ == خالی_سوچک) {
#جے_تعریف_شدہ GGML_BACKEND_DIR
        تلاش_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#جے_انت
        // default search paths: executable directory, current directory
        تلاش_paths.push_back(get_executable_رستہ());
        std::error_code cwd_ec;
        ثابت fs::path cwd = fs::current_path(cwd_ec);
        إذا (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.سنیہا().c_str());
        } وإلا {
            تلاش_paths.push_back(cwd);
        }
    } وإلا {
        تلاش_paths.push_back(fs::u8path(user_تلاش_رستہ));
    }

    صحيح بہترین_score = 0;
    fs::path بہترین_رستہ;
    std::error_code ec;

    لكل (ثابت خودکار & تلاش_رستہ : تلاش_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(تلاش_رستہ, dir_ec);

        لكل (ثابت fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ثابت خودکار & اندراج = *dir_it;
            إذا (اندراج.is_regular_file(ec)) {
                خودکار filename = اندراج.path().filename();
                خودکار ext = اندراج.path().extension();
                إذا (filename.native().find(file_اگیتر) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_لدو_library(اندراج) };
                    إذا (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, رستہ_str(اندراج.path()).c_str(), dl_error());
                    }
                    إذا (handle) {
                        خودکار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        إذا (score_fn) {
                            صحيح s = score_fn();
#جے_بنا_تعریف NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, رستہ_str(اندراج.path()).c_str(), s);
#جے_انت
                            إذا (s > بہترین_score) {
                                بہترین_score = s;
                                بہترین_رستہ = اندراج.path();
                            }
                        } وإلا {
                            إذا (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, رستہ_str(اندراج.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        إذا (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, رستہ_str(تلاش_رستہ).c_str(), dir_ec.سنیہا().c_str());
        }
    }

    إذا (بہترین_score == 0) {
        // try to load the base backend
        لكل (ثابت خودکار & تلاش_رستہ : تلاش_paths) {
            fs::path filename = backend_filename_اگیتر().native() + name_رستہ.native() + backend_filename_extension().native();
            fs::path path = تلاش_رستہ / filename;
            إذا (std::error_code ec; fs::exists(path, ec)) {
                أرجع get_reg().لدو_backend(path, silent);
            } وإلا {
                إذا (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, رستہ_str(path).c_str(), ec.سنیہا().c_str());
                }
            }
        }
        أرجع خالی_سوچک;
    }

    أرجع get_reg().لدو_backend(بہترین_رستہ, silent);
}

فراغ ggml_backend_لدو_all() {
    ggml_backend_لدو_all_from_رستہ(خالی_سوچک);
}

فراغ ggml_backend_لدو_all_from_رستہ(ثابت حرف * dir_رستہ) {
#جے_تعریف_شدہ NDEBUG
    منطقي silent = سچ;
#وإلا
    منطقي silent = خطأ;
#جے_انت

    ggml_backend_لدو_بہترین("blas", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("zendnn", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("cann", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("cuda", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("hip", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("metal", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("rpc", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("sycl", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("vulkan", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("virtgpu", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("opencl", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("hexagon", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("musa", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("openvino", silent, dir_رستہ);
    ggml_backend_لدو_بہترین("cpu", silent, dir_رستہ);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ثابت حرف * backend_رستہ = std::getenv("GGML_BACKEND_PATH");
    إذا (backend_رستہ) {
        ggml_backend_لدو(backend_رستہ);
    }
}
