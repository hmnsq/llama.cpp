#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "model-cache.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/properties.hpp>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <optional>

ov::Core & ov_singleton_core() {
    ಸ್ಥಿರ ov::Core core;
    ಹಿಂತಿರುಗಿ core;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ has_ಪೂರ್ವಪ್ರತ್ಯಯ(ಸ್ಥಿರಾಂಕ std::string & s, ಸ್ಥಿರಾಂಕ std::string & prefix) {
    ಹಿಂತಿರುಗಿ s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_virtual_routing_device(ಸ್ಥಿರಾಂಕ std::string & device_name) {
    ಹಿಂತಿರುಗಿ has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "AUTO") || has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "MULTI") || has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "HETERO");
}

ಸ್ಥಿರ std::vector<std::string> ov_enumerate_devices() {
    std::vector<std::string> ಫಲಿತಾಂಶ;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & device : ov_singleton_core().get_ಲಭ್ಯ_devices()) {
        ಆದರೆ (!is_virtual_routing_device(device)) {
            ಫಲಿತಾಂಶ.push_back(device);
        }
    }

    ಆದರೆ (ಫಲಿತಾಂಶ.ಖಾಲಿಗಳು()) {
        ಫಲಿತಾಂಶ.push_back("CPU");
    }

    std::sort(ಫಲಿತಾಂಶ.begin(), ಫಲಿತಾಂಶ.end());
    ಫಲಿತಾಂಶ.erase(std::unique(ಫಲಿತಾಂಶ.begin(), ಫಲಿತಾಂಶ.end()), ಫಲಿತಾಂಶ.end());
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

std::string ggml_openvino_get_device_description(ಸ್ಥಿರಾಂಕ std::string & device_name) {
    std::string description = device_name;
    ಪ್ರಯತ್ನಿಸು {
        description = ov_singleton_core().get_property(device_name, ov::device::ತುಂಬಿದ_name);
    } ಹಿಡಿ (...) {
        ಹಿಂತಿರುಗಿ device_name;
    }

    ಆದರೆ (has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "NPU")) {
        ಪ್ರಯತ್ನಿಸು {
            ಸ್ಥಿರಾಂಕ std::string arch = ov_singleton_core().get_property(device_name, "DEVICE_ARCHITECTURE").as<std::string>();
            ಆದರೆ (!arch.ಖಾಲಿಗಳು()) {
                description += " (NPU " + arch + ")";
            }
        } ಹಿಡಿ (...) {
        }
    }

    ಹಿಂತಿರುಗಿ description;
}

// requested: GGML_OPENVINO_DEVICE, nullptr if unset. available_devices is never empty (see ov_enumerate_devices)
ಸ್ಥಿರ std::string resolve_openvino_device_name(ಸ್ಥಿರಾಂಕ std::vector<std::string> & ಲಭ್ಯ_devices,
                                                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * requested) {
    ಸ್ವಯಂಚಾಲಿತ ಲಭ್ಯ = [&](ಸ್ಥಿರಾಂಕ std::string & name) {
        ಹಿಂತಿರುಗಿ std::find(ಲಭ್ಯ_devices.begin(), ಲಭ್ಯ_devices.end(), name) != ಲಭ್ಯ_devices.end();
    };
    ಆದರೆ (requested == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಲಭ್ಯ("CPU") ? "CPU" : ಲಭ್ಯ_devices.front();
    }
    ಆದರೆ (!ಲಭ್ಯ(requested)) {
        // No fallback to CPU (easy to miss) and no GPU -> GPU.0 alias (with iGPU + dGPU, GPU.0 is often the
        // wrong one). List the devices here: --list-devices initializes this backend and would abort too.
        std::string list;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ std::string & name : ಲಭ್ಯ_devices) {
            list += "\n  " + name + ": " + ggml_openvino_get_device_description(name);
        }
        GGML_ABORT("GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s is not available. "
                   "Set it to one of the available OpenVINO devices:%s",
                   requested, list.c_str());
    }
    ಹಿಂತಿರುಗಿ requested;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

ಶೂನ್ಯ ggml_openvino_device_config::init() {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);
    ಆದರೆ (initialized) {
        ಹಿಂತಿರುಗಿ;
    }
    // Set up front: a failed OpenCL setup below is not retried on every call
    initialized = ನಿಜ;

    // All recognized GGML_OPENVINO_* env vars. Their values are cached here
    // once at backend init time and read back via ggml_openvino_getenv_str()
    // (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
    ಸ್ಥಿರ constexpr ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * env_var_ಹೆಸರುಗಳು[] = {
        // String values (use ggml_openvino_getenv_str)
        "GGML_OPENVINO_DEVICE",
        "GGML_OPENVINO_CACHE_DIR",
        "GGML_OPENVINO_SPILL_DIR",
        "GGML_OPENVINO_DEBUG_NODE",
        "GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR",
        "GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY",
        "GGML_OPENVINO_NPU_COMPILE_CONFIG",
        // Integer values (use ggml_openvino_getenv_int)
        "GGML_OPENVINO_PREFILL_CHUNK_SIZE",
        // Boolean toggles (treated as int flags via ggml_openvino_getenv_int)
        "GGML_OPENVINO_STATEFUL_EXECUTION",
        "GGML_OPENVINO_PROFILING",
        "GGML_OPENVINO_DUMP_CGRAPH",
        "GGML_OPENVINO_DUMP_IR",
        "GGML_OPENVINO_DEBUG_INPUT",
        "GGML_OPENVINO_DEBUG_OUTPUT",
        // Force the static (NPU-shape) compute path on any device, e.g. GGML_OPENVINO_DEVICE=CPU,
        // to test the static-shape translation without NPUW/real NPU hardware in the loop.
        "GGML_OPENVINO_FORCE_STATIC",
        "GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS",
        "GGML_OPENVINO_ENABLE_CACHE",
        "GGML_OPENVINO_DISABLE_CACHE",
        "GGML_OPENVINO_DISABLE_KV_SLICE",
        "GGML_OPENVINO_ENABLE_FALLBACK",
        "GGML_OPENVINO_MANUAL_GQA_ATTN",
        "GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN",
        "GGML_OPENVINO_MOE_OP",
        "GGML_OPENVINO_MEMORY_OPTIMIZE",
        "GGML_OPENVINO_RELEASE_WEIGHTS",
        "GGML_OPENVINO_REDUCE_COMPILE_MEM",
        "GGML_OPENVINO_LOG_UNSUPPORTED_OPS",
        "GGML_OPENVINO_LOG_SWA_LAYERS",
        "GGML_OPENVINO_NATIVE_SOFTPLUS",
        "GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS",
        "GGML_OPENVINO_REQUANT_KQUANT",
        "GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT",
        // Build the precise (but O(n_nodes)) graph cache key. Needed by op tests.
        "GGML_OPENVINO_FULL_GRAPH_KEY",
    };

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ & env_var : env_var_ಹೆಸರುಗಳು) {
        ಸ್ವಯಂಚಾಲಿತ * env = getenv(env_var);
        ಆದರೆ (env) {
            environment_variables[env_var] = env;
        }
    }

    ಲಭ್ಯ_devices = ov_enumerate_devices();
    device_name = resolve_openvino_device_name(ಲಭ್ಯ_devices, ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE"));
    is_npu = has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "NPU");

    ggml_openvino_model_cache_init();

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    ಆದರೆ (has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "NPU")) {
        compile_config = {
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        ಆದರೆ (cache_dir && ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_ಬಹುಲಕ(ov::Cache_ಬಹುಲಕ::OPTIMIZE_ಗಾತ್ರ));
        }
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * compilation_ಬಹುಲಕ_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        ಆದರೆ (compilation_ಬಹುಲಕ_params && ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(compilation_ಬಹುಲಕ_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_ಬಹುಲಕ_params;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (cache_dir && ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_ಬಹುಲಕ(ov::Cache_ಬಹುಲಕ::OPTIMIZE_ಗಾತ್ರ));
    }

    ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2) {
        compile_config.insert(ov::enable_profiling(ನಿಜ));
    }

    // Initialize remote context with queue sharing for GPU
    ಆದರೆ (has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "GPU")) {
        // Use the OpenCL context OpenVINO created for this device, so GPU.N gets its own device
        cl_context cl_ctx;
        ಪ್ರಯತ್ನಿಸು {
            ಸ್ವಯಂಚಾಲಿತ ov_ctx = ov_singleton_core().get_default_context(device_name).as<ov::intel_gpu::ocl::ClContext>();
            cl_ctx = ov_ctx.get();
        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
            // The consumers of the remote context have no host fallback, and OpenVINO
            // already reported the device as present.
            GGML_ABORT("ggml-openvino: failed to get the OpenCL context for %s: %s", device_name.c_str(), e.what());
        }

        cl_int err;
        cl_device_id cl_device;
        err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, ಗಾತ್ರ(cl_device), &cl_device, ಶೂನ್ಯಸೂಚಕ);
        ಆದರೆ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL device for %s: %d", device_name.c_str(), err);
        }

        cl_platform_id cl_platform;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_PLATFORM, ಗಾತ್ರ(cl_platform), &cl_platform, ಶೂನ್ಯಸೂಚಕ);
        ಆದರೆ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL platform for %s: %d", device_name.c_str(), err);
        }

        cl_mem_fill_fn =
            (cl_ಸರತಿಗೆ_ಸೇರಿಸು_Mem_Fill_INTEL_fn) cl_Get_Extension_Function_ವಿಳಾಸ_For_Platform(cl_platform, "clEnqueueMemFillINTEL");
        cl_mem_cpy_fn =
            (cl_ಸರತಿಗೆ_ಸೇರಿಸು_Memcpy_INTEL_fn) cl_Get_Extension_Function_ವಿಳಾಸ_For_Platform(cl_platform, "clEnqueueMemcpyINTEL");

        cl_ulong device_max_alloc = 0;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_MAX_MEM_ALLOC_ಗಾತ್ರ, ಗಾತ್ರ(device_max_alloc), &device_max_alloc,
                              ಶೂನ್ಯಸೂಚಕ);
        ಆದರೆ (err == CL_SUCCESS) {
            max_alloc_ಗಾತ್ರ = device_max_alloc;
        } ಇಲ್ಲದಿದ್ದರೆ {
            // not fatal, ggml then allocates one buffer
            GGML_LOG_WARN("Failed to get OpenCL max allocation size: %d\n", err);
        }

        ಸ್ಥಿರಾಂಕ cl_ಸರತಿ_properties profiling_properties[] = {
            CL_ಸರತಿ_PROPERTIES,
            CL_ಸರತಿ_PROFILING_ENABLE,
            0,
        };
        ಸ್ಥಿರಾಂಕ cl_ಸರತಿ_properties * ಸರತಿ_properties =
            ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? profiling_properties : ಶೂನ್ಯಸೂಚಕ;
        cl_ಸರತಿ = cl_Create_Command_ಸರತಿ_With_Properties(cl_ctx, cl_device, ಸರತಿ_properties, &err);
        ಆದರೆ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to create the OpenCL queue for %s: %d", device_name.c_str(), err);
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_ಸರತಿ);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (has_ಪೂರ್ವಪ್ರತ್ಯಯ(device_name, "NPU")) {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    ಆದರೆ (cl_ಸರತಿ != ಶೂನ್ಯಸೂಚಕ) {
        cl_Release_Command_ಸರತಿ(cl_ಸರತಿ);
        cl_ಸರತಿ = ಶೂನ್ಯಸೂಚಕ;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    ಸ್ಥಿರ ggml_openvino_device_config config;
    ಹಿಂತಿರುಗಿ config;
}

// Initialize device config (call during backend init)
ಶೂನ್ಯ ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
ಸ್ಥಿರಾಂಕ std::string & ggml_openvino_get_device_name() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().device_name;
}

std::vector<std::string> ggml_openvino_get_ಲಭ್ಯ_devices() {
    ಸ್ವಯಂಚಾಲಿತ & config = ggml_openvino_get_device_config();
    config.init();
    ಹಿಂತಿರುಗಿ config.ಲಭ್ಯ_devices;
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_openvino_getenv_str(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * var, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * default_ಮೌಲ್ಯ) {
    ಸ್ವಯಂಚಾಲಿತ & env_map = ggml_openvino_get_device_config().environment_variables;
    ಸ್ವಯಂಚಾಲಿತ it = env_map.find(var);
    ಹಿಂತಿರುಗಿ (it == env_map.end() || it->second.ಖಾಲಿಗಳು()) ? default_ಮೌಲ್ಯ : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int (via std::atoi).
// Returns default_value (0) when the var is unset or empty. Used for both
// integer settings (e.g. GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean
// toggles: "0" disables, any non-zero integer enables.
ಪೂರ್ಣಾಂಕ ggml_openvino_getenv_int(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * var, ಪೂರ್ಣಾಂಕ default_ಮೌಲ್ಯ) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v = ggml_openvino_getenv_str(var, ಶೂನ್ಯಸೂಚಕ);
    ಹಿಂತಿರುಗಿ v ? std::atoi(v) : default_ಮೌಲ್ಯ;
}

ತಾರ್ಕಿಕ ggml_openvino_reduce_compile_mem_enabled() {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    ಆದರೆ (reduce_compile_mem != ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    ಹಿಂತಿರುಗಿ ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

ತಾರ್ಕಿಕ ggml_openvino_release_weights_enabled() {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    ಆದರೆ (release_weights != ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    ಹಿಂತಿರುಗಿ ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
ತಾರ್ಕಿಕ ggml_openvino_is_npu() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().is_npu;
}

ತಾರ್ಕಿಕ ggml_openvino_is_gpu() {
    ಹಿಂತಿರುಗಿ has_ಪೂರ್ವಪ್ರತ್ಯಯ(ggml_openvino_get_device_name(), "GPU");
}

size_t ggml_openvino_max_alloc_ಗಾತ್ರ() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().max_alloc_ಗಾತ್ರ;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
ಸ್ಥಿರಾಂಕ ov::AnyMap & ggml_openvino_get_compile_config() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_ಸರತಿ ggml_openvino_get_cl_ಸರತಿ() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().cl_ಸರತಿ;
}

// Get the clEnqueueMemFillINTEL function pointer
cl_ಸರತಿಗೆ_ಸೇರಿಸು_Mem_Fill_INTEL_fn ggml_openvino_get_cl_ಸರತಿಗೆ_ಸೇರಿಸು_Mem_Fill_INTEL() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().cl_mem_fill_fn;
}

// Get the clEnqueueMemcpyINTEL function pointer
cl_ಸರತಿಗೆ_ಸೇರಿಸು_Memcpy_INTEL_fn ggml_openvino_get_cl_ಸರತಿಗೆ_ಸೇರಿಸು_Memcpy_INTEL() {
    ಹಿಂತಿರುಗಿ ggml_openvino_get_device_config().cl_mem_cpy_fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ತಾರ್ಕಿಕ no_requant) {
    ಆದರೆ (no_requant) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(tensor->name, "token_embd.weight", 17) == 0) {
        ಹಿಂತಿರುಗಿ ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 :
                                                                             ExtraQuantType::Q8_0_C);
    }
    ಆದರೆ (ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(tensor->name, "output.weight", 13) == 0) {
        ಹಿಂತಿರುಗಿ ExtraQuantType::Q8_0_C;
    }
    ಆದರೆ (ggml_openvino_is_npu()) {
        ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_0_128;
    }
    // By default Q6_K/Q5_K are requantized to Q8_0_C, which *inflates* 6- and 5-bit weights to 8
    // while the rest of the model stays at 4 bits, and Q4_K keeps its native group-32 layout
    // (an f16 scale plus an f16 zero point per 32 weights = 0.125 B/weight of metadata).
    // Decode of a large model is bandwidth-bound, so both cost throughput.
    //
    // GGML_OPENVINO_REQUANT_KQUANT selects a 4-bit target instead. Names are
    // q4_<sym|asym><group>[_all]: <sym|asym> says whether a per-group zero point is kept, <group>
    // is the group size, and the _all suffix sends Q4_K down the same path (without it only
    // Q6_K/Q5_K are touched):
    //   q4_sym128      Q6_K/Q5_K -> Q4_0_128 (u4, group 128, symmetric)
    //   q4_sym128_all  and Q4_K too -- drops Q4_K's per-32 zero point, which costs some accuracy
    //   q4_asym64      Q6_K/Q5_K -> Q4_1_64 (u4, group 64, asymmetric)
    //   q4_asym64_all  Q6_K/Q5_K and Q4_K -> Q4_1_64 (u4, group 64, asymmetric)
    //   native         no requantization at all (keep Q6_K/Q5_K as they are)
    //
    // q4_asym64 leaves Q4_K at its native group 32. Use q4_asym64_all to keep the group size uniform.
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * rq = ggml_openvino_getenv_str("GGML_OPENVINO_REQUANT_KQUANT");
    ಸ್ವಯಂಚಾಲಿತ is_opt = [rq](ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
        ಹಿಂತಿರುಗಿ rq && strcmp(rq, name) == 0;
    };
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sym128 = is_opt("q4_sym128");
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sym128_all = is_opt("q4_sym128_all");
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ asym64_all = is_opt("q4_asym64_all");
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ asym64 = is_opt("q4_asym64");

    ಆದರೆ (tensor->type == GGML_TYPE_Q4_K) {
        ಆದರೆ (sym128_all) {
            ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_0_128;
        }
        ಆದರೆ (asym64_all) {
            ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_1_64;
        }
    }
    // MoE expert weights (3D, ne[2] = n_expert) stored as Q5_1/Q8_0 are the expert-side
    // equivalent of Q6_K/Q5_K: kept at 8 bits by default while the rest of the model is at 4
    // (gemma-4 26B-A4B keeps its down projection there). Send them to 4 bits under the same
    // option, at group 64 rather than 128: the down expert has k=704, which 64 divides
    // (704/64 = 11) and 128 does not.
    ಆದರೆ (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        ಆದರೆ (sym128 || sym128_all) {
            ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_0_64;
        }
        ಆದರೆ (asym64 || asym64_all) {
            ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_1_64;
        }
        // TODO: temporary workaround for a known OpenVINO GPU-plugin bug -- remove once the
        // plugin computes grouped 8-bit GatherMatmulCompressed correctly. This costs accuracy
        // (5/8-bit -> 4-bit) on any model it applies to, so it must not outlive the bug.
        //
        // On GPU these would otherwise stay in their native *grouped 8-bit* layout, which the GPU
        // plugin's GatherMatmulCompressed computes incorrectly -- gemma-4 26B-A4B (whose down
        // projection is Q5_1) produces garbage, while the same graph is correct on CPU. It is
        // specific to grouped 8 bit: the gate/up experts are grouped u4 *with* a zero point and
        // are fine, and Qwen3.5 / granite are fine because their Q5_K/Q6_K down projections
        // already requantize to per-channel Q8_0_C (grouped=0). Sending these to grouped 4 bit
        // avoids the broken layout and restores correct output.
        // Opt out with GGML_OPENVINO_REQUANT_KQUANT=native.
        ಆದರೆ (ggml_openvino_is_gpu() && !is_opt("native")) {
            ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_0_64;
        }
    }
    ಆಯ್ಕೆ (tensor->type) {
    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಆದರೆ (sym128 || sym128_all) {
            ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_0_128;
        }
        ಆದರೆ (asym64 || asym64_all) {
            ಹಿಂತಿರುಗಿ ExtraQuantType::Q4_1_64;
        }
        ಆದರೆ (is_opt("native")) {
            ಹಿಂತಿರುಗಿ std::nullopt;
        }
        ಹಿಂತಿರುಗಿ ExtraQuantType::Q8_0_C;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ತಾರ್ಕಿಕ use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_ಸಮಮಿತ = ಸುಳ್ಳು;

    ಆದರೆ (!ggml_is_quantized(tensor->type)) {
        ಹಿಂತಿರುಗಿ layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    ಆದರೆ (tensor->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ layout;
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    ಸ್ಥಿರಾಂಕ size_t alignment = 64;  // Good for SIMD

    ಆದರೆ (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_ಪ್ರತಿ_ಖಂಡ = 32;
        layout.is_ಸಮಮಿತ = ನಿಜ;
        layout.weights_ಗಾತ್ರ = ggml_nbytes(tensor);
        layout.weights_ಸ್ಥಾನಾಂತರ = 0;
        layout.ಒಟ್ಟು_ಗಾತ್ರ = layout.weights_ಗಾತ್ರ;
        ಹಿಂತಿರುಗಿ layout;
    }

    // Check if requantization is needed (NPU-specific)
    ಸ್ವಯಂಚಾಲಿತ requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    ಆದರೆ (requant_type.has_value()) {
        layout.is_requant = ನಿಜ;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        ಆದರೆ (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_ಗಾತ್ರ = n_elements * ಗಾತ್ರ(uint16_t);  // F16 = 2 bytes
            layout.ಒಟ್ಟು_ಗಾತ್ರ = layout.weights_ಗಾತ್ರ;
            layout.weights_ಸ್ಥಾನಾಂತರ = 0;
            // No scales/zp for F16
            ಹಿಂತಿರುಗಿ layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        ಆಯ್ಕೆ (requant_type.value()) {
        ಸ್ಥಿತಿ ExtraQuantType::Q4_0_128:
            layout.is_u4 = ನಿಜ;
            layout.weights_ಪ್ರತಿ_ಖಂಡ = 128;
            layout.is_ಸಮಮಿತ = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ExtraQuantType::Q4_1_64:
            layout.is_u4 = ನಿಜ;
            layout.weights_ಪ್ರತಿ_ಖಂಡ = 64;
            layout.is_ಸಮಮಿತ = ಸುಳ್ಳು;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ExtraQuantType::Q4_0_64:
            layout.is_u4 = ನಿಜ;
            layout.weights_ಪ್ರತಿ_ಖಂಡ = 64;
            layout.is_ಸಮಮಿತ = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ExtraQuantType::Q4_0_C:
            layout.is_u4 = ನಿಜ;
            layout.weights_ಪ್ರತಿ_ಖಂಡ = tensor->ne[0];
            layout.is_ಸಮಮಿತ = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ExtraQuantType::Q8_0_32:
            layout.is_u4 = ಸುಳ್ಳು;
            layout.weights_ಪ್ರತಿ_ಖಂಡ = 32;
            layout.is_ಸಮಮಿತ = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ExtraQuantType::Q8_0_C:
            layout.is_u4 = ಸುಳ್ಳು;
            layout.weights_ಪ್ರತಿ_ಖಂಡ = tensor->ne[0];
            layout.is_ಸಮಮಿತ = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ExtraQuantType::Q8_1_C:
            layout.is_u4 = ಸುಳ್ಳು;
            layout.weights_ಪ್ರತಿ_ಖಂಡ = tensor->ne[0];
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }

        ಆದರೆ (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_ಗಾತ್ರ = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_ಖಂಡಗಳು = n_elements / layout.weights_ಪ್ರತಿ_ಖಂಡ;
            layout.scales_ಗಾತ್ರ = n_ಖಂಡಗಳು * ಗಾತ್ರ(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            ಆದರೆ (layout.is_ಸಮಮಿತ) {
                layout.zp_ಗಾತ್ರ = 0;
            } ಇಲ್ಲದಿದ್ದರೆ {
                layout.zp_ಗಾತ್ರ = layout.is_u4 ? ((n_ಖಂಡಗಳು + 1) / 2) : n_ಖಂಡಗಳು;
            }

            layout.weights_ಸ್ಥಾನಾಂತರ = 0;
            layout.scales_ಸ್ಥಾನಾಂತರ = ((layout.weights_ಗಾತ್ರ + alignment - 1) / alignment) * alignment;
            layout.zp_ಸ್ಥಾನಾಂತರ = layout.scales_ಸ್ಥಾನಾಂತರ + ((layout.scales_ಗಾತ್ರ + alignment - 1) / alignment) * alignment;
            layout.ಒಟ್ಟು_ಗಾತ್ರ = layout.zp_ಸ್ಥಾನಾಂತರ + layout.zp_ಗಾತ್ರ;
            layout.ಒಟ್ಟು_ಗಾತ್ರ = std::max(layout.ಒಟ್ಟು_ಗಾತ್ರ, ggml_nbytes(tensor));
            ಹಿಂತಿರುಗಿ layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = ಸುಳ್ಳು;
    layout.weights_ಪ್ರತಿ_ಖಂಡ = 32;
    layout.is_ಸಮಮಿತ = ಸುಳ್ಳು;

    ಆಯ್ಕೆ (tensor->type) {
    ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        layout.is_u4 = ನಿಜ;
        layout.is_ಸಮಮಿತ = ನಿಜ;
        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        layout.is_u4 = ನಿಜ;
        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        layout.is_ಸಮಮಿತ = ನಿಜ;
        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        layout.weights_ಪ್ರತಿ_ಖಂಡ = 16;
        layout.is_ಸಮಮಿತ = ನಿಜ;
        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ನಿಲ್ಲಿಸು;

    ಪೂರ್ವನಿಯೋಜಿತ:
        // Unsupported quantization type
        ಹಿಂತಿರುಗಿ layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_ಗಾತ್ರ = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_ಖಂಡಗಳು = n_elements / layout.weights_ಪ್ರತಿ_ಖಂಡ;
    layout.scales_ಗಾತ್ರ = n_ಖಂಡಗಳು * (tensor->type == GGML_TYPE_MXFP4 ? ಗಾತ್ರ(uint8_t) : ಗಾತ್ರ(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    ಆದರೆ (layout.is_ಸಮಮಿತ) {
        layout.zp_ಗಾತ್ರ = 0;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_bias || for_gather_matmul) {
        layout.zp_ಗಾತ್ರ = n_ಖಂಡಗಳು * ಗಾತ್ರ(uint16_t);
    } ಇಲ್ಲದಿದ್ದರೆ {
        layout.zp_ಗಾತ್ರ = layout.is_u4 ? ((n_ಖಂಡಗಳು + 1) / 2) : n_ಖಂಡಗಳು;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_ಸ್ಥಾನಾಂತರ = 0;
    layout.scales_ಸ್ಥಾನಾಂತರ = ((layout.weights_ಗಾತ್ರ + alignment - 1) / alignment) * alignment;
    layout.zp_ಸ್ಥಾನಾಂತರ = layout.scales_ಸ್ಥಾನಾಂತರ + ((layout.scales_ಗಾತ್ರ + alignment - 1) / alignment) * alignment;
    layout.ಒಟ್ಟು_ಗಾತ್ರ = layout.zp_ಸ್ಥಾನಾಂತರ + layout.zp_ಗಾತ್ರ;
    layout.ಒಟ್ಟು_ಗಾತ್ರ = std::max(layout.ಒಟ್ಟು_ಗಾತ್ರ, ggml_nbytes(tensor));

    ಹಿಂತಿರುಗಿ layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ತಾರ್ಕಿಕ is_remote) {
    ov::Shape shape;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    ಆಯ್ಕೆ (tensor->type) {
    ಸ್ಥಿತಿ GGML_TYPE_F32:
        element_type = ov::element::f32;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_F16:
        element_type = ov::element::f16;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_I32:
        element_type = ov::element::i32;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_I64:
        element_type = ov::element::i64;
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಸ್ವಯಂಚಾಲಿತ remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    ಆದರೆ (is_remote) {
        GGML_ASSERT(ggml_openvino_is_gpu());
        ಸ್ವಯಂಚಾಲಿತ gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        ಸ್ವಯಂಚಾಲಿತ usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->ದತ್ತಾಂಶ);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->ದತ್ತಾಂಶ);
    }

    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_openvino_tensor_extra(ov_tensor);
}
