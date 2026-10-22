#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino-extra.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "model-cache.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/properties.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>

ov::Core & ov_singleton_core() {
    ਸਥਿਰ ov::Core core;
    ਵਾਪਸ core;
}

ਸਥਿਰ ਤਾਰਕਿਕ has_ਅਗੇਤਰ(ਅਚਲ std::string & s, ਅਚਲ std::string & prefix) {
    ਵਾਪਸ s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

ਸਥਿਰ ਤਾਰਕਿਕ is_virtual_routing_device(ਅਚਲ std::string & device_name) {
    ਵਾਪਸ has_ਅਗੇਤਰ(device_name, "AUTO") || has_ਅਗੇਤਰ(device_name, "MULTI") || has_ਅਗੇਤਰ(device_name, "HETERO");
}

ਸਥਿਰ std::vector<std::string> ov_enumerate_devices() {
    std::vector<std::string> ਨਤੀਜਾ;

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & device : ov_singleton_core().get_ਉਪਲਬਧ_devices()) {
        ਜੇ (!is_virtual_routing_device(device)) {
            ਨਤੀਜਾ.push_back(device);
        }
    }

    ਜੇ (ਨਤੀਜਾ.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਨਤੀਜਾ.push_back("CPU");
    }

    std::sort(ਨਤੀਜਾ.begin(), ਨਤੀਜਾ.end());
    ਨਤੀਜਾ.erase(std::unique(ਨਤੀਜਾ.begin(), ਨਤੀਜਾ.end()), ਨਤੀਜਾ.end());
    ਵਾਪਸ ਨਤੀਜਾ;
}

std::string ggml_openvino_get_device_description(ਅਚਲ std::string & device_name) {
    std::string description = device_name;
    ਕੋਸ਼ਿਸ਼ {
        description = ov_singleton_core().get_property(device_name, ov::device::ਭਰਿਆ_name);
    } ਫੜੋ (...) {
        ਵਾਪਸ device_name;
    }

    ਜੇ (has_ਅਗੇਤਰ(device_name, "NPU")) {
        ਕੋਸ਼ਿਸ਼ {
            ਅਚਲ std::string arch = ov_singleton_core().get_property(device_name, "DEVICE_ARCHITECTURE").as<std::string>();
            ਜੇ (!arch.ਖਾਲੀ_ਥਾਵਾਂ()) {
                description += " (NPU " + arch + ")";
            }
        } ਫੜੋ (...) {
        }
    }

    ਵਾਪਸ description;
}

// requested: GGML_OPENVINO_DEVICE, nullptr if unset. available_devices is never empty (see ov_enumerate_devices)
ਸਥਿਰ std::string resolve_openvino_device_name(ਅਚਲ std::vector<std::string> & ਉਪਲਬਧ_devices,
                                                ਅਚਲ ਅੱਖਰ * requested) {
    ਸਵੈਚਾਲਿਤ ਉਪਲਬਧ = [&](ਅਚਲ std::string & name) {
        ਵਾਪਸ std::find(ਉਪਲਬਧ_devices.begin(), ਉਪਲਬਧ_devices.end(), name) != ਉਪਲਬਧ_devices.end();
    };
    ਜੇ (requested == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਉਪਲਬਧ("CPU") ? "CPU" : ਉਪਲਬਧ_devices.front();
    }
    ਜੇ (!ਉਪਲਬਧ(requested)) {
        // No fallback to CPU (easy to miss) and no GPU -> GPU.0 alias (with iGPU + dGPU, GPU.0 is often the
        // wrong one). List the devices here: --list-devices initializes this backend and would abort too.
        std::string list;
        ਲਈ (ਅਚਲ std::string & name : ਉਪਲਬਧ_devices) {
            list += "\n  " + name + ": " + ggml_openvino_get_device_description(name);
        }
        GGML_ABORT("GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s is not available. "
                   "Set it to one of the available OpenVINO devices:%s",
                   requested, list.c_str());
    }
    ਵਾਪਸ requested;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

ਖਾਲੀ ggml_openvino_device_config::init() {
    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
    ਜੇ (initialized) {
        ਵਾਪਸ;
    }
    // Set up front: a failed OpenCL setup below is not retried on every call
    initialized = ਸੱਚ;

    // All recognized GGML_OPENVINO_* env vars. Their values are cached here
    // once at backend init time and read back via ggml_openvino_getenv_str()
    // (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
    ਸਥਿਰ constexpr ਅਚਲ ਅੱਖਰ * env_var_ਨਾਂ[] = {
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

    ਲਈ (ਅਚਲ ਅੱਖਰ * ਅਚਲ & env_var : env_var_ਨਾਂ) {
        ਸਵੈਚਾਲਿਤ * env = getenv(env_var);
        ਜੇ (env) {
            environment_variables[env_var] = env;
        }
    }

    ਉਪਲਬਧ_devices = ov_enumerate_devices();
    device_name = resolve_openvino_device_name(ਉਪਲਬਧ_devices, ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE"));
    is_npu = has_ਅਗੇਤਰ(device_name, "NPU");

    ggml_openvino_model_cache_init();

    ਅਚਲ ਅੱਖਰ * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    ਜੇ (has_ਅਗੇਤਰ(device_name, "NPU")) {
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
        ਜੇ (cache_dir && ਲੜੀ_ਲੰਬਾਈ(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_ਬਹੁਲਕ(ov::Cache_ਬਹੁਲਕ::OPTIMIZE_ਮਾਪ));
        }
        ਅਚਲ ਅੱਖਰ * compilation_ਬਹੁਲਕ_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        ਜੇ (compilation_ਬਹੁਲਕ_params && ਲੜੀ_ਲੰਬਾਈ(compilation_ਬਹੁਲਕ_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_ਬਹੁਲਕ_params;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (cache_dir && ਲੜੀ_ਲੰਬਾਈ(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_ਬਹੁਲਕ(ov::Cache_ਬਹੁਲਕ::OPTIMIZE_ਮਾਪ));
    }

    ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2) {
        compile_config.insert(ov::enable_profiling(ਸੱਚ));
    }

    // Initialize remote context with queue sharing for GPU
    ਜੇ (has_ਅਗੇਤਰ(device_name, "GPU")) {
        // Use the OpenCL context OpenVINO created for this device, so GPU.N gets its own device
        cl_context cl_ctx;
        ਕੋਸ਼ਿਸ਼ {
            ਸਵੈਚਾਲਿਤ ov_ctx = ov_singleton_core().get_default_context(device_name).as<ov::intel_gpu::ocl::ClContext>();
            cl_ctx = ov_ctx.get();
        } ਫੜੋ (ਅਚਲ std::exception & e) {
            // The consumers of the remote context have no host fallback, and OpenVINO
            // already reported the device as present.
            GGML_ABORT("ggml-openvino: failed to get the OpenCL context for %s: %s", device_name.c_str(), e.what());
        }

        cl_int err;
        cl_device_id cl_device;
        err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, ਆਕਾਰ(cl_device), &cl_device, ਖਾਲੀ_ਸੂਚਕ);
        ਜੇ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL device for %s: %d", device_name.c_str(), err);
        }

        cl_platform_id cl_platform;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_PLATFORM, ਆਕਾਰ(cl_platform), &cl_platform, ਖਾਲੀ_ਸੂਚਕ);
        ਜੇ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL platform for %s: %d", device_name.c_str(), err);
        }

        cl_mem_fill_fn =
            (cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Mem_Fill_INTEL_fn) cl_Get_Extension_Function_ਪਤਾ_For_Platform(cl_platform, "clEnqueueMemFillINTEL");
        cl_mem_cpy_fn =
            (cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Memcpy_INTEL_fn) cl_Get_Extension_Function_ਪਤਾ_For_Platform(cl_platform, "clEnqueueMemcpyINTEL");

        cl_ulong device_max_alloc = 0;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_MAX_MEM_ALLOC_ਮਾਪ, ਆਕਾਰ(device_max_alloc), &device_max_alloc,
                              ਖਾਲੀ_ਸੂਚਕ);
        ਜੇ (err == CL_SUCCESS) {
            max_alloc_ਮਾਪ = device_max_alloc;
        } ਨਹੀਂਤਾਂ {
            // not fatal, ggml then allocates one buffer
            GGML_LOG_WARN("Failed to get OpenCL max allocation size: %d\n", err);
        }

        ਅਚਲ cl_ਕਤਾਰ_properties profiling_properties[] = {
            CL_ਕਤਾਰ_PROPERTIES,
            CL_ਕਤਾਰ_PROFILING_ENABLE,
            0,
        };
        ਅਚਲ cl_ਕਤਾਰ_properties * ਕਤਾਰ_properties =
            ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? profiling_properties : ਖਾਲੀ_ਸੂਚਕ;
        cl_ਕਤਾਰ = cl_Create_Command_ਕਤਾਰ_With_Properties(cl_ctx, cl_device, ਕਤਾਰ_properties, &err);
        ਜੇ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to create the OpenCL queue for %s: %d", device_name.c_str(), err);
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_ਕਤਾਰ);
    } ਨਹੀਂਤਾਂ ਜੇ (has_ਅਗੇਤਰ(device_name, "NPU")) {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    ਜੇ (cl_ਕਤਾਰ != ਖਾਲੀ_ਸੂਚਕ) {
        cl_Release_Command_ਕਤਾਰ(cl_ਕਤਾਰ);
        cl_ਕਤਾਰ = ਖਾਲੀ_ਸੂਚਕ;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    ਸਥਿਰ ggml_openvino_device_config config;
    ਵਾਪਸ config;
}

// Initialize device config (call during backend init)
ਖਾਲੀ ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
ਅਚਲ std::string & ggml_openvino_get_device_name() {
    ਵਾਪਸ ggml_openvino_get_device_config().device_name;
}

std::vector<std::string> ggml_openvino_get_ਉਪਲਬਧ_devices() {
    ਸਵੈਚਾਲਿਤ & config = ggml_openvino_get_device_config();
    config.init();
    ਵਾਪਸ config.ਉਪਲਬਧ_devices;
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
ਅਚਲ ਅੱਖਰ * ggml_openvino_getenv_str(ਅਚਲ ਅੱਖਰ * var, ਅਚਲ ਅੱਖਰ * default_ਮੁੱਲ) {
    ਸਵੈਚਾਲਿਤ & env_map = ggml_openvino_get_device_config().environment_variables;
    ਸਵੈਚਾਲਿਤ it = env_map.find(var);
    ਵਾਪਸ (it == env_map.end() || it->second.ਖਾਲੀ_ਥਾਵਾਂ()) ? default_ਮੁੱਲ : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int (via std::atoi).
// Returns default_value (0) when the var is unset or empty. Used for both
// integer settings (e.g. GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean
// toggles: "0" disables, any non-zero integer enables.
ਪੂਰਨ ggml_openvino_getenv_int(ਅਚਲ ਅੱਖਰ * var, ਪੂਰਨ default_ਮੁੱਲ) {
    ਅਚਲ ਅੱਖਰ * v = ggml_openvino_getenv_str(var, ਖਾਲੀ_ਸੂਚਕ);
    ਵਾਪਸ v ? std::atoi(v) : default_ਮੁੱਲ;
}

ਤਾਰਕਿਕ ggml_openvino_reduce_compile_mem_enabled() {
    ਅਚਲ ਅੱਖਰ * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    ਜੇ (reduce_compile_mem != ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    ਵਾਪਸ ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

ਤਾਰਕਿਕ ggml_openvino_release_weights_enabled() {
    ਅਚਲ ਅੱਖਰ * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    ਜੇ (release_weights != ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    ਵਾਪਸ ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
ਤਾਰਕਿਕ ggml_openvino_is_npu() {
    ਵਾਪਸ ggml_openvino_get_device_config().is_npu;
}

ਤਾਰਕਿਕ ggml_openvino_is_gpu() {
    ਵਾਪਸ has_ਅਗੇਤਰ(ggml_openvino_get_device_name(), "GPU");
}

size_t ggml_openvino_max_alloc_ਮਾਪ() {
    ਵਾਪਸ ggml_openvino_get_device_config().max_alloc_ਮਾਪ;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    ਵਾਪਸ ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
ਅਚਲ ov::AnyMap & ggml_openvino_get_compile_config() {
    ਵਾਪਸ ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_ਕਤਾਰ ggml_openvino_get_cl_ਕਤਾਰ() {
    ਵਾਪਸ ggml_openvino_get_device_config().cl_ਕਤਾਰ;
}

// Get the clEnqueueMemFillINTEL function pointer
cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Mem_Fill_INTEL_fn ggml_openvino_get_cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Mem_Fill_INTEL() {
    ਵਾਪਸ ggml_openvino_get_device_config().cl_mem_fill_fn;
}

// Get the clEnqueueMemcpyINTEL function pointer
cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Memcpy_INTEL_fn ggml_openvino_get_cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Memcpy_INTEL() {
    ਵਾਪਸ ggml_openvino_get_device_config().cl_mem_cpy_fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ਅਚਲ ggml_tensor * tensor, ਤਾਰਕਿਕ no_requant) {
    ਜੇ (no_requant) {
        ਵਾਪਸ std::nullopt;
    }
    ਜੇ (ਲੜੀ_ਤੁਲਨਾ(tensor->name, "token_embd.weight", 17) == 0) {
        ਵਾਪਸ ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 :
                                                                             ExtraQuantType::Q8_0_C);
    }
    ਜੇ (ਲੜੀ_ਤੁਲਨਾ(tensor->name, "output.weight", 13) == 0) {
        ਵਾਪਸ ExtraQuantType::Q8_0_C;
    }
    ਜੇ (ggml_openvino_is_npu()) {
        ਵਾਪਸ ExtraQuantType::Q4_0_128;
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
    ਅਚਲ ਅੱਖਰ * rq = ggml_openvino_getenv_str("GGML_OPENVINO_REQUANT_KQUANT");
    ਸਵੈਚਾਲਿਤ is_opt = [rq](ਅਚਲ ਅੱਖਰ * name) {
        ਵਾਪਸ rq && strcmp(rq, name) == 0;
    };
    ਅਚਲ ਤਾਰਕਿਕ sym128 = is_opt("q4_sym128");
    ਅਚਲ ਤਾਰਕਿਕ sym128_all = is_opt("q4_sym128_all");
    ਅਚਲ ਤਾਰਕਿਕ asym64_all = is_opt("q4_asym64_all");
    ਅਚਲ ਤਾਰਕਿਕ asym64 = is_opt("q4_asym64");

    ਜੇ (tensor->type == GGML_TYPE_Q4_K) {
        ਜੇ (sym128_all) {
            ਵਾਪਸ ExtraQuantType::Q4_0_128;
        }
        ਜੇ (asym64_all) {
            ਵਾਪਸ ExtraQuantType::Q4_1_64;
        }
    }
    // MoE expert weights (3D, ne[2] = n_expert) stored as Q5_1/Q8_0 are the expert-side
    // equivalent of Q6_K/Q5_K: kept at 8 bits by default while the rest of the model is at 4
    // (gemma-4 26B-A4B keeps its down projection there). Send them to 4 bits under the same
    // option, at group 64 rather than 128: the down expert has k=704, which 64 divides
    // (704/64 = 11) and 128 does not.
    ਜੇ (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        ਜੇ (sym128 || sym128_all) {
            ਵਾਪਸ ExtraQuantType::Q4_0_64;
        }
        ਜੇ (asym64 || asym64_all) {
            ਵਾਪਸ ExtraQuantType::Q4_1_64;
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
        ਜੇ (ggml_openvino_is_gpu() && !is_opt("native")) {
            ਵਾਪਸ ExtraQuantType::Q4_0_64;
        }
    }
    ਚੋਣ (tensor->type) {
    ਹਾਲਤ GGML_TYPE_Q6_K:
    ਹਾਲਤ GGML_TYPE_Q5_K:
        ਜੇ (sym128 || sym128_all) {
            ਵਾਪਸ ExtraQuantType::Q4_0_128;
        }
        ਜੇ (asym64 || asym64_all) {
            ਵਾਪਸ ExtraQuantType::Q4_1_64;
        }
        ਜੇ (is_opt("native")) {
            ਵਾਪਸ std::nullopt;
        }
        ਵਾਪਸ ExtraQuantType::Q8_0_C;
    ਮੂਲ:
        ਵਾਪਸ std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(ਅਚਲ ggml_tensor * tensor, ਤਾਰਕਿਕ use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_ਸਮਮਿਤ = ਝੂਠ;

    ਜੇ (!ggml_is_quantized(tensor->type)) {
        ਵਾਪਸ layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    ਜੇ (tensor->ne[3] != 1) {
        ਵਾਪਸ layout;
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    ਅਚਲ ਤਾਰਕਿਕ for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    ਅਚਲ size_t alignment = 64;  // Good for SIMD

    ਜੇ (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = 32;
        layout.is_ਸਮਮਿਤ = ਸੱਚ;
        layout.weights_ਮਾਪ = ggml_nbytes(tensor);
        layout.weights_ਵਿੱਥ = 0;
        layout.ਕੁੱਲ_ਮਾਪ = layout.weights_ਮਾਪ;
        ਵਾਪਸ layout;
    }

    // Check if requantization is needed (NPU-specific)
    ਸਵੈਚਾਲਿਤ requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    ਜੇ (requant_type.has_value()) {
        layout.is_requant = ਸੱਚ;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        ਜੇ (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_ਮਾਪ = n_elements * ਆਕਾਰ(uint16_t);  // F16 = 2 bytes
            layout.ਕੁੱਲ_ਮਾਪ = layout.weights_ਮਾਪ;
            layout.weights_ਵਿੱਥ = 0;
            // No scales/zp for F16
            ਵਾਪਸ layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        ਚੋਣ (requant_type.value()) {
        ਹਾਲਤ ExtraQuantType::Q4_0_128:
            layout.is_u4 = ਸੱਚ;
            layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = 128;
            layout.is_ਸਮਮਿਤ = ਸੱਚ;
            ਰੋਕੋ;
        ਹਾਲਤ ExtraQuantType::Q4_1_64:
            layout.is_u4 = ਸੱਚ;
            layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = 64;
            layout.is_ਸਮਮਿਤ = ਝੂਠ;
            ਰੋਕੋ;
        ਹਾਲਤ ExtraQuantType::Q4_0_64:
            layout.is_u4 = ਸੱਚ;
            layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = 64;
            layout.is_ਸਮਮਿਤ = ਸੱਚ;
            ਰੋਕੋ;
        ਹਾਲਤ ExtraQuantType::Q4_0_C:
            layout.is_u4 = ਸੱਚ;
            layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = tensor->ne[0];
            layout.is_ਸਮਮਿਤ = ਸੱਚ;
            ਰੋਕੋ;
        ਹਾਲਤ ExtraQuantType::Q8_0_32:
            layout.is_u4 = ਝੂਠ;
            layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = 32;
            layout.is_ਸਮਮਿਤ = ਸੱਚ;
            ਰੋਕੋ;
        ਹਾਲਤ ExtraQuantType::Q8_0_C:
            layout.is_u4 = ਝੂਠ;
            layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = tensor->ne[0];
            layout.is_ਸਮਮਿਤ = ਸੱਚ;
            ਰੋਕੋ;
        ਹਾਲਤ ExtraQuantType::Q8_1_C:
            layout.is_u4 = ਝੂਠ;
            layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = tensor->ne[0];
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }

        ਜੇ (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_ਮਾਪ = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_ਖੰਡਾਂ = n_elements / layout.weights_ਪ੍ਰਤੀ_ਖੰਡ;
            layout.scales_ਮਾਪ = n_ਖੰਡਾਂ * ਆਕਾਰ(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            ਜੇ (layout.is_ਸਮਮਿਤ) {
                layout.zp_ਮਾਪ = 0;
            } ਨਹੀਂਤਾਂ {
                layout.zp_ਮਾਪ = layout.is_u4 ? ((n_ਖੰਡਾਂ + 1) / 2) : n_ਖੰਡਾਂ;
            }

            layout.weights_ਵਿੱਥ = 0;
            layout.scales_ਵਿੱਥ = ((layout.weights_ਮਾਪ + alignment - 1) / alignment) * alignment;
            layout.zp_ਵਿੱਥ = layout.scales_ਵਿੱਥ + ((layout.scales_ਮਾਪ + alignment - 1) / alignment) * alignment;
            layout.ਕੁੱਲ_ਮਾਪ = layout.zp_ਵਿੱਥ + layout.zp_ਮਾਪ;
            layout.ਕੁੱਲ_ਮਾਪ = std::max(layout.ਕੁੱਲ_ਮਾਪ, ggml_nbytes(tensor));
            ਵਾਪਸ layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = ਝੂਠ;
    layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = 32;
    layout.is_ਸਮਮਿਤ = ਝੂਠ;

    ਚੋਣ (tensor->type) {
    ਹਾਲਤ GGML_TYPE_MXFP4:
    ਹਾਲਤ GGML_TYPE_Q4_0:
        layout.is_u4 = ਸੱਚ;
        layout.is_ਸਮਮਿਤ = ਸੱਚ;
        ਰੋਕੋ;

    ਹਾਲਤ GGML_TYPE_Q4_1:
    ਹਾਲਤ GGML_TYPE_Q4_K:
        layout.is_u4 = ਸੱਚ;
        ਰੋਕੋ;

    ਹਾਲਤ GGML_TYPE_Q8_0:
        layout.is_ਸਮਮਿਤ = ਸੱਚ;
        ਰੋਕੋ;

    ਹਾਲਤ GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        ਰੋਕੋ;

    ਹਾਲਤ GGML_TYPE_Q6_K:
        layout.weights_ਪ੍ਰਤੀ_ਖੰਡ = 16;
        layout.is_ਸਮਮਿਤ = ਸੱਚ;
        ਰੋਕੋ;

    ਹਾਲਤ GGML_TYPE_Q5_K:
        ਰੋਕੋ;

    ਮੂਲ:
        // Unsupported quantization type
        ਵਾਪਸ layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_ਮਾਪ = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_ਖੰਡਾਂ = n_elements / layout.weights_ਪ੍ਰਤੀ_ਖੰਡ;
    layout.scales_ਮਾਪ = n_ਖੰਡਾਂ * (tensor->type == GGML_TYPE_MXFP4 ? ਆਕਾਰ(uint8_t) : ਆਕਾਰ(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    ਜੇ (layout.is_ਸਮਮਿਤ) {
        layout.zp_ਮਾਪ = 0;
    } ਨਹੀਂਤਾਂ ਜੇ (use_bias || for_gather_matmul) {
        layout.zp_ਮਾਪ = n_ਖੰਡਾਂ * ਆਕਾਰ(uint16_t);
    } ਨਹੀਂਤਾਂ {
        layout.zp_ਮਾਪ = layout.is_u4 ? ((n_ਖੰਡਾਂ + 1) / 2) : n_ਖੰਡਾਂ;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_ਵਿੱਥ = 0;
    layout.scales_ਵਿੱਥ = ((layout.weights_ਮਾਪ + alignment - 1) / alignment) * alignment;
    layout.zp_ਵਿੱਥ = layout.scales_ਵਿੱਥ + ((layout.scales_ਮਾਪ + alignment - 1) / alignment) * alignment;
    layout.ਕੁੱਲ_ਮਾਪ = layout.zp_ਵਿੱਥ + layout.zp_ਮਾਪ;
    layout.ਕੁੱਲ_ਮਾਪ = std::max(layout.ਕੁੱਲ_ਮਾਪ, ggml_nbytes(tensor));

    ਵਾਪਸ layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(ਅਚਲ ggml_tensor * tensor, ਤਾਰਕਿਕ is_remote) {
    ov::Shape shape;
    ਲਈ (ਪੂਰਨ i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    ਚੋਣ (tensor->type) {
    ਹਾਲਤ GGML_TYPE_F32:
        element_type = ov::element::f32;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_F16:
        element_type = ov::element::f16;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_I32:
        element_type = ov::element::i32;
        ਰੋਕੋ;
    ਹਾਲਤ GGML_TYPE_I64:
        element_type = ov::element::i64;
        ਰੋਕੋ;
    ਮੂਲ:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਸਵੈਚਾਲਿਤ remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    ਜੇ (is_remote) {
        GGML_ASSERT(ggml_openvino_is_gpu());
        ਸਵੈਚਾਲਿਤ gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        ਸਵੈਚਾਲਿਤ usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->ਅੰਕੜੇ);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } ਨਹੀਂਤਾਂ {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->ਅੰਕੜੇ);
    }

    ਵਾਪਸ ਨਵਾਂ ggml_openvino_tensor_extra(ov_tensor);
}
