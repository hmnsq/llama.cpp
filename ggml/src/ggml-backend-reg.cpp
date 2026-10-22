#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-backend-dl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstring>
#అన్తర్భావ <filesystem>
#అన్తర్భావ <memory>
#అన్తర్భావ <string>
#అన్తర్భావ <type_traits>
#అన్తర్భావ <vector>
#అన్తర్భావ <cctype>

#నిర్వచితమైతే _WIN32
#    define WIN32_LEAN_AND_సగటు
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#లేదావేళ defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#లేకపోతే
#    include <dlfcn.h>
#    include <unistd.h>
#అయితే_ముగింపు

// Backend registry
#నిర్వచితమైతే GGML_USE_CPU
#అన్తర్భావ "ggml-cpu.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CUDA
#అన్తర్భావ "ggml-cuda.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_METAL
#అన్తర్భావ "ggml-metal.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_SYCL
#అన్తర్భావ "ggml-sycl.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_VULKAN
#అన్తర్భావ "ggml-vulkan.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_WEBGPU
#అన్తర్భావ "ggml-webgpu.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_ZDNN
#అన్తర్భావ "ggml-zdnn.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_OPENCL
#అన్తర్భావ "ggml-opencl.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_HEXAGON
#అన్తర్భావ "ggml-hexagon.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_BLAS
#అన్తర్భావ "ggml-blas.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_RPC
#అన్తర్భావ "ggml-rpc.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_VIRTGPU_FRONTEND
#అన్తర్భావ "ggml-virtgpu.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_CANN
#అన్తర్భావ "ggml-cann.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_ZENDNN
#అన్తర్భావ "ggml-zendnn.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_OPENVINO
#అన్తర్భావ "ggml-openvino.h"
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_ET
#అన్తర్భావ "ggml-et.h"
#అయితే_ముగింపు

నామాకాశ fs = std::filesystem;

స్థిర std::string మార్గం_str(స్థిరరాశి fs::path & path) {
    ప్రయత్నించు {
#ఒకవేళ defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        స్థిరరాశి std::u8string u8str = path.u8string();
        తిరిగి std::string(reinterpret_cast<స్థిరరాశి అక్షరం *>(u8str.దత్తాంశం()), u8str.size());
#లేకపోతే
        // C++17: u8string() returns std::string
        తిరిగి path.u8string();
#అయితే_ముగింపు
    } పట్టుకో (...) {
        తిరిగి std::string();
    }
}

నిర్మాణం ggml_backend_reg_నమోదు {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

నిర్మాణం ggml_backend_registry {
    std::vector<ggml_backend_reg_నమోదు> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#నిర్వచితమైతే GGML_USE_CUDA
        రిజిస్టరు_backend(ggml_backend_cuda_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_METAL
        రిజిస్టరు_backend(ggml_backend_metal_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_SYCL
        రిజిస్టరు_backend(ggml_backend_sycl_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_VULKAN
    // Add runtime disable check
    ఒకవేళ (getenv("GGML_DISABLE_VULKAN") == శూన్యసూచిక) {
        రిజిస్టరు_backend(ggml_backend_vk_reg());
    } లేకపోతే {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_WEBGPU
        రిజిస్టరు_backend(ggml_backend_webgpu_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_ZDNN
        రిజిస్టరు_backend(ggml_backend_zdnn_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_VIRTGPU_FRONTEND
        రిజిస్టరు_backend(ggml_backend_virtgpu_reg());
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_USE_OPENCL
        రిజిస్టరు_backend(ggml_backend_opencl_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_ZENDNN
        రిజిస్టరు_backend(ggml_backend_zendnn_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_HEXAGON
        రిజిస్టరు_backend(ggml_backend_hexagon_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_CANN
        రిజిస్టరు_backend(ggml_backend_cann_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_BLAS
        రిజిస్టరు_backend(ggml_backend_blas_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_RPC
        రిజిస్టరు_backend(ggml_backend_rpc_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_OPENVINO
        రిజిస్టరు_backend(ggml_backend_openvino_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_ET
        రిజిస్టరు_backend(ggml_backend_et_reg());
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_USE_CPU
        రిజిస్టరు_backend(ggml_backend_cpu_reg());
#అయితే_ముగింపు
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        కొరకు (స్వయంచాలక & నమోదు_2 : backends) {
            ఒకవేళ (నమోదు_2.handle) {
                నమోదు_2.handle.release(); // NOLINT
            }
        }
    }

    శూన్య రిజిస్టరు_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = శూన్యసూచిక) {
        ఒకవేళ (!reg) {
            తిరిగి;
        }

        కొరకు (స్వయంచాలక & నమోదు_2 : backends) {
            ఒకవేళ (నమోదు_2.reg == reg) {
                తిరిగి;
            }
        }

#నిర్వచితం_కాకపోతే NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_లెక్క(reg));
#అయితే_ముగింపు
        backends.push_back({ reg, std::move(handle) });
        కొరకు (size_t i = 0; i < ggml_backend_reg_dev_లెక్క(reg); i++) {
            రిజిస్టరు_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    శూన్య రిజిస్టరు_device(ggml_backend_dev_t device) {
        కొరకు (స్వయంచాలక & dev : devices) {
            ఒకవేళ (dev == device) {
                తిరిగి;
            }
        }

#నిర్వచితం_కాకపోతే NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#అయితే_ముగింపు
        devices.push_back(device);
    }

    ggml_backend_reg_t ఎక్కించు_backend(స్థిరరాశి fs::path & path, తార్కిక silent) {
        dl_handle_ptr handle { dl_ఎక్కించు_library(path) };
        ఒకవేళ (!handle) {
            ఒకవేళ (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, మార్గం_str(path).c_str(), dl_error());
            }
            తిరిగి శూన్యసూచిక;
        }

        స్వయంచాలక score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        ఒకవేళ (score_fn && score_fn() == 0) {
            ఒకవేళ (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, మార్గం_str(path).c_str());
            }
            తిరిగి శూన్యసూచిక;
        }

        స్వయంచాలక backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        ఒకవేళ (!backend_init_fn) {
            ఒకవేళ (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, మార్గం_str(path).c_str());
            }
            తిరిగి శూన్యసూచిక;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        ఒకవేళ (!reg || reg->api_సంస్కరణ != GGML_BACKEND_API_సంస్కరణ) {
            ఒకవేళ (!silent) {
                ఒకవేళ (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, మార్గం_str(path).c_str());
                } లేకపోతే {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, మార్గం_str(path).c_str(), reg->api_సంస్కరణ, GGML_BACKEND_API_సంస్కరణ);
                }
            }
            తిరిగి శూన్యసూచిక;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), మార్గం_str(path).c_str());

        రిజిస్టరు_backend(reg, std::move(handle));

        తిరిగి reg;
    }

    శూన్య unload_backend(ggml_backend_reg_t reg, తార్కిక silent) {
        స్వయంచాలక it = std::find_if(backends.begin(), backends.end(),
                               [reg](స్థిరరాశి ggml_backend_reg_నమోదు & నమోదు_2) { తిరిగి నమోదు_2.reg == reg; });

        ఒకవేళ (it == backends.end()) {
            ఒకవేళ (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            తిరిగి;
        }

        ఒకవేళ (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { తిరిగి ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

స్థిర ggml_backend_registry & get_reg() {
    స్థిర ggml_backend_registry reg;
    తిరిగి reg;
}

// Internal API
శూన్య ggml_backend_రిజిస్టరు(ggml_backend_reg_t reg) {
    get_reg().రిజిస్టరు_backend(reg);
}

శూన్య ggml_backend_device_రిజిస్టరు(ggml_backend_dev_t device) {
    get_reg().రిజిస్టరు_device(device);
}

// Backend (reg) enumeration
స్థిర తార్కిక striequals(స్థిరరాశి అక్షరం * a, స్థిరరాశి అక్షరం * b) {
    కొరకు (; *a && *b; a++, b++) {
        ఒకవేళ (std::tolower(*a) != std::tolower(*b)) {
            తిరిగి అబద్ధం;
        }
    }
    తిరిగి *a == *b;
}

size_t ggml_backend_reg_లెక్క() {
    తిరిగి get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t సూచీ) {
    GGML_ASSERT(సూచీ < ggml_backend_reg_లెక్క());
    తిరిగి get_reg().backends[సూచీ].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(స్థిరరాశి అక్షరం * name) {
    కొరకు (size_t i = 0; i < ggml_backend_reg_లెక్క(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        ఒకవేళ (striequals(ggml_backend_reg_name(reg), name)) {
            తిరిగి reg;
        }
    }
    తిరిగి శూన్యసూచిక;
}

// Device enumeration
size_t ggml_backend_dev_లెక్క() {
    తిరిగి get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t సూచీ) {
    GGML_ASSERT(సూచీ < ggml_backend_dev_లెక్క());
    తిరిగి get_reg().devices[సూచీ];
}

ggml_backend_dev_t ggml_backend_dev_by_name(స్థిరరాశి అక్షరం * name) {
    కొరకు (size_t i = 0; i < ggml_backend_dev_లెక్క(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ఒకవేళ (striequals(ggml_backend_dev_name(dev), name)) {
            తిరిగి dev;
        }
    }
    తిరిగి శూన్యసూచిక;
}

ggml_backend_dev_t ggml_backend_dev_by_type(గణన ggml_backend_dev_type type) {
    కొరకు (size_t i = 0; i < ggml_backend_dev_లెక్క(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ఒకవేళ (ggml_backend_dev_type(dev) == type) {
            తిరిగి dev;
        }
    }
    తిరిగి శూన్యసూచిక;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(స్థిరరాశి అక్షరం * name, స్థిరరాశి అక్షరం * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    ఒకవేళ (!dev) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(గణన ggml_backend_dev_type type, స్థిరరాశి అక్షరం * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    ఒకవేళ (!dev) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_ఉత్తమ(శూన్య) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ఒకవేళ (!dev) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి ggml_backend_dev_init(dev, శూన్యసూచిక);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_ఎక్కించు(స్థిరరాశి అక్షరం * path) {
    తిరిగి get_reg().ఎక్కించు_backend(path, అబద్ధం);
}

శూన్య ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, నిజం);
}

స్థిర fs::path get_executable_మార్గం() {
#ఒకవేళ defined(__APPLE__)
    // get executable path
    std::vector<అక్షరం> path;
    uint32_t size;
    అయితే (నిజం) {
        size = path.size();
        ఒకవేళ (NSGet_Executable_మార్గం(path.దత్తాంశం(), &size) == 0) {
            ఆపు;
        }
        path.resize(size);
    }
    std::string పునాది_మార్గం(path.దత్తాంశం(), size);
    // remove executable name
    స్వయంచాలక చివరి_slash = పునాది_మార్గం.find_last_of('/');
    ఒకవేళ (చివరి_slash != std::string::npos) {
        పునాది_మార్గం = పునాది_మార్గం.substr(0, చివరి_slash);
    }
    తిరిగి పునాది_మార్గం + "/";
#లేదావేళ defined(__linux__) || defined(__FreeBSD__)
    std::string పునాది_మార్గం = ".";
    std::vector<అక్షరం> path(1024);
    అయితే (నిజం) {
        // get executable path
#    ఒకవేళ defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.దత్తాంశం(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.దత్తాంశం(), path.size());
#    endif
        ఒకవేళ (len == -1) {
            ఆపు;
        }
        ఒకవేళ (len < (ssize_t) path.size()) {
            పునాది_మార్గం = std::string(path.దత్తాంశం(), len);
            // remove executable name
            స్వయంచాలక చివరి_slash = పునాది_మార్గం.find_last_of('/');
            ఒకవేళ (చివరి_slash != std::string::npos) {
                పునాది_మార్గం = పునాది_మార్గం.substr(0, చివరి_slash);
            }
            ఆపు;
        }
        path.resize(path.size() * 2);
    }

    తిరిగి పునాది_మార్గం + "/";
#లేదావేళ defined(_WIN32)
    std::vector<wchar_t> path(MAX_మార్గం);
    DWORD len = GetModuleFileNameW(ఖాళీ, path.దత్తాంశం(), path.size());
    ఒకవేళ (len == 0) {
        తిరిగి {};
    }
    std::wstring పునాది_మార్గం(path.దత్తాంశం(), len);
    // remove executable name
    స్వయంచాలక చివరి_slash = పునాది_మార్గం.find_last_of('\\');
    ఒకవేళ (చివరి_slash != std::string::npos) {
        పునాది_మార్గం = పునాది_మార్గం.substr(0, చివరి_slash);
    }
    తిరిగి పునాది_మార్గం + L"\\";
#లేకపోతే
    తిరిగి {};
#అయితే_ముగింపు
}

స్థిర fs::path backend_filename_ఉపసర్గ() {
#నిర్వచితమైతే _WIN32
    తిరిగి fs::u8path("ggml-");
#లేకపోతే
    తిరిగి fs::u8path("libggml-");
#అయితే_ముగింపు
}

స్థిర fs::path backend_filename_extension() {
#నిర్వచితమైతే _WIN32
    తిరిగి fs::u8path(".dll");
#లేకపోతే
    తిరిగి fs::u8path(".so");
#అయితే_ముగింపు
}

స్థిర ggml_backend_reg_t ggml_backend_ఎక్కించు_ఉత్తమ(స్థిరరాశి అక్షరం * name, తార్కిక silent, స్థిరరాశి అక్షరం * user_శోధన_మార్గం) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    స్థిరరాశి fs::path name_మార్గం = fs::u8path(name);
    స్థిరరాశి fs::path file_ఉపసర్గ = backend_filename_ఉపసర్గ().native() + name_మార్గం.native() + fs::u8path("-").native();
    స్థిరరాశి fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> శోధన_paths;
    ఒకవేళ (user_శోధన_మార్గం == శూన్యసూచిక) {
#నిర్వచితమైతే GGML_BACKEND_DIR
        శోధన_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#అయితే_ముగింపు
        // default search paths: executable directory, current directory
        శోధన_paths.push_back(get_executable_మార్గం());
        std::error_code cwd_ec;
        స్థిరరాశి fs::path cwd = fs::current_path(cwd_ec);
        ఒకవేళ (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.సందేశం().c_str());
        } లేకపోతే {
            శోధన_paths.push_back(cwd);
        }
    } లేకపోతే {
        శోధన_paths.push_back(fs::u8path(user_శోధన_మార్గం));
    }

    పూర్ణాంకం ఉత్తమ_score = 0;
    fs::path ఉత్తమ_మార్గం;
    std::error_code ec;

    కొరకు (స్థిరరాశి స్వయంచాలక & శోధన_మార్గం : శోధన_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(శోధన_మార్గం, dir_ec);

        కొరకు (స్థిరరాశి fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            స్థిరరాశి స్వయంచాలక & నమోదు_2 = *dir_it;
            ఒకవేళ (నమోదు_2.is_regular_file(ec)) {
                స్వయంచాలక filename = నమోదు_2.path().filename();
                స్వయంచాలక ext = నమోదు_2.path().extension();
                ఒకవేళ (filename.native().find(file_ఉపసర్గ) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_ఎక్కించు_library(నమోదు_2) };
                    ఒకవేళ (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, మార్గం_str(నమోదు_2.path()).c_str(), dl_error());
                    }
                    ఒకవేళ (handle) {
                        స్వయంచాలక score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        ఒకవేళ (score_fn) {
                            పూర్ణాంకం s = score_fn();
#నిర్వచితం_కాకపోతే NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, మార్గం_str(నమోదు_2.path()).c_str(), s);
#అయితే_ముగింపు
                            ఒకవేళ (s > ఉత్తమ_score) {
                                ఉత్తమ_score = s;
                                ఉత్తమ_మార్గం = నమోదు_2.path();
                            }
                        } లేకపోతే {
                            ఒకవేళ (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, మార్గం_str(నమోదు_2.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        ఒకవేళ (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, మార్గం_str(శోధన_మార్గం).c_str(), dir_ec.సందేశం().c_str());
        }
    }

    ఒకవేళ (ఉత్తమ_score == 0) {
        // try to load the base backend
        కొరకు (స్థిరరాశి స్వయంచాలక & శోధన_మార్గం : శోధన_paths) {
            fs::path filename = backend_filename_ఉపసర్గ().native() + name_మార్గం.native() + backend_filename_extension().native();
            fs::path path = శోధన_మార్గం / filename;
            ఒకవేళ (std::error_code ec; fs::exists(path, ec)) {
                తిరిగి get_reg().ఎక్కించు_backend(path, silent);
            } లేకపోతే {
                ఒకవేళ (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, మార్గం_str(path).c_str(), ec.సందేశం().c_str());
                }
            }
        }
        తిరిగి శూన్యసూచిక;
    }

    తిరిగి get_reg().ఎక్కించు_backend(ఉత్తమ_మార్గం, silent);
}

శూన్య ggml_backend_ఎక్కించు_all() {
    ggml_backend_ఎక్కించు_all_from_మార్గం(శూన్యసూచిక);
}

శూన్య ggml_backend_ఎక్కించు_all_from_మార్గం(స్థిరరాశి అక్షరం * dir_మార్గం) {
#నిర్వచితమైతే NDEBUG
    తార్కిక silent = నిజం;
#లేకపోతే
    తార్కిక silent = అబద్ధం;
#అయితే_ముగింపు

    ggml_backend_ఎక్కించు_ఉత్తమ("blas", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("zendnn", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("cann", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("cuda", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("hip", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("metal", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("rpc", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("sycl", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("vulkan", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("virtgpu", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("opencl", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("hexagon", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("musa", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("openvino", silent, dir_మార్గం);
    ggml_backend_ఎక్కించు_ఉత్తమ("cpu", silent, dir_మార్గం);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    స్థిరరాశి అక్షరం * backend_మార్గం = std::getenv("GGML_BACKEND_PATH");
    ఒకవేళ (backend_మార్గం) {
        ggml_backend_ఎక్కించు(backend_మార్గం);
    }
}
