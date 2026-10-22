#అన్తర్భావ "ggml-openvino-extra.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "model-cache.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#అన్తర్భావ <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#అన్తర్భావ <openvino/runtime/properties.hpp>
#అన్తర్భావ <mutex>
#అన్తర్భావ <optional>

ov::Core & ov_singleton_core() {
    స్థిర ov::Core core;
    తిరిగి core;
}

స్థిర తార్కిక has_ఉపసర్గ(స్థిరరాశి std::string & s, స్థిరరాశి std::string & prefix) {
    తిరిగి s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

స్థిర తార్కిక is_virtual_routing_device(స్థిరరాశి std::string & device_name) {
    తిరిగి has_ఉపసర్గ(device_name, "AUTO") || has_ఉపసర్గ(device_name, "MULTI") || has_ఉపసర్గ(device_name, "HETERO");
}

స్థిర std::vector<std::string> ov_enumerate_devices() {
    std::vector<std::string> ఫలితం;

    కొరకు (స్థిరరాశి స్వయంచాలక & device : ov_singleton_core().get_అందుబాటు_devices()) {
        ఒకవేళ (!is_virtual_routing_device(device)) {
            ఫలితం.push_back(device);
        }
    }

    ఒకవేళ (ఫలితం.ఖాళీలు()) {
        ఫలితం.push_back("CPU");
    }

    std::sort(ఫలితం.begin(), ఫలితం.end());
    ఫలితం.erase(std::unique(ఫలితం.begin(), ఫలితం.end()), ఫలితం.end());
    తిరిగి ఫలితం;
}

std::string ggml_openvino_get_device_description(స్థిరరాశి std::string & device_name) {
    std::string description = device_name;
    ప్రయత్నించు {
        description = ov_singleton_core().get_property(device_name, ov::device::నిండిన_name);
    } పట్టుకో (...) {
        తిరిగి device_name;
    }

    ఒకవేళ (has_ఉపసర్గ(device_name, "NPU")) {
        ప్రయత్నించు {
            స్థిరరాశి std::string arch = ov_singleton_core().get_property(device_name, "DEVICE_ARCHITECTURE").as<std::string>();
            ఒకవేళ (!arch.ఖాళీలు()) {
                description += " (NPU " + arch + ")";
            }
        } పట్టుకో (...) {
        }
    }

    తిరిగి description;
}

// requested: GGML_OPENVINO_DEVICE, nullptr if unset. available_devices is never empty (see ov_enumerate_devices)
స్థిర std::string resolve_openvino_device_name(స్థిరరాశి std::vector<std::string> & అందుబాటు_devices,
                                                స్థిరరాశి అక్షరం * requested) {
    స్వయంచాలక అందుబాటు = [&](స్థిరరాశి std::string & name) {
        తిరిగి std::find(అందుబాటు_devices.begin(), అందుబాటు_devices.end(), name) != అందుబాటు_devices.end();
    };
    ఒకవేళ (requested == శూన్యసూచిక) {
        తిరిగి అందుబాటు("CPU") ? "CPU" : అందుబాటు_devices.front();
    }
    ఒకవేళ (!అందుబాటు(requested)) {
        // No fallback to CPU (easy to miss) and no GPU -> GPU.0 alias (with iGPU + dGPU, GPU.0 is often the
        // wrong one). List the devices here: --list-devices initializes this backend and would abort too.
        std::string list;
        కొరకు (స్థిరరాశి std::string & name : అందుబాటు_devices) {
            list += "\n  " + name + ": " + ggml_openvino_get_device_description(name);
        }
        GGML_ABORT("GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s is not available. "
                   "Set it to one of the available OpenVINO devices:%s",
                   requested, list.c_str());
    }
    తిరిగి requested;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

శూన్య ggml_openvino_device_config::init() {
    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);
    ఒకవేళ (initialized) {
        తిరిగి;
    }
    // Set up front: a failed OpenCL setup below is not retried on every call
    initialized = నిజం;

    // All recognized GGML_OPENVINO_* env vars. Their values are cached here
    // once at backend init time and read back via ggml_openvino_getenv_str()
    // (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
    స్థిర constexpr స్థిరరాశి అక్షరం * env_var_పేర్లు[] = {
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

    కొరకు (స్థిరరాశి అక్షరం * స్థిరరాశి & env_var : env_var_పేర్లు) {
        స్వయంచాలక * env = getenv(env_var);
        ఒకవేళ (env) {
            environment_variables[env_var] = env;
        }
    }

    అందుబాటు_devices = ov_enumerate_devices();
    device_name = resolve_openvino_device_name(అందుబాటు_devices, ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE"));
    is_npu = has_ఉపసర్గ(device_name, "NPU");

    ggml_openvino_model_cache_init();

    స్థిరరాశి అక్షరం * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    ఒకవేళ (has_ఉపసర్గ(device_name, "NPU")) {
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
        ఒకవేళ (cache_dir && అక్షరశ్రేణి_పొడవు(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_బహుళకం(ov::Cache_బహుళకం::OPTIMIZE_కొలత));
        }
        స్థిరరాశి అక్షరం * compilation_బహుళకం_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        ఒకవేళ (compilation_బహుళకం_params && అక్షరశ్రేణి_పొడవు(compilation_బహుళకం_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_బహుళకం_params;
        }
    } లేకపోతే ఒకవేళ (cache_dir && అక్షరశ్రేణి_పొడవు(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_బహుళకం(ov::Cache_బహుళకం::OPTIMIZE_కొలత));
    }

    ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2) {
        compile_config.insert(ov::enable_profiling(నిజం));
    }

    // Initialize remote context with queue sharing for GPU
    ఒకవేళ (has_ఉపసర్గ(device_name, "GPU")) {
        // Use the OpenCL context OpenVINO created for this device, so GPU.N gets its own device
        cl_context cl_ctx;
        ప్రయత్నించు {
            స్వయంచాలక ov_ctx = ov_singleton_core().get_default_context(device_name).as<ov::intel_gpu::ocl::ClContext>();
            cl_ctx = ov_ctx.get();
        } పట్టుకో (స్థిరరాశి std::exception & e) {
            // The consumers of the remote context have no host fallback, and OpenVINO
            // already reported the device as present.
            GGML_ABORT("ggml-openvino: failed to get the OpenCL context for %s: %s", device_name.c_str(), e.what());
        }

        cl_int err;
        cl_device_id cl_device;
        err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, పరిమాణం(cl_device), &cl_device, శూన్యసూచిక);
        ఒకవేళ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL device for %s: %d", device_name.c_str(), err);
        }

        cl_platform_id cl_platform;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_PLATFORM, పరిమాణం(cl_platform), &cl_platform, శూన్యసూచిక);
        ఒకవేళ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL platform for %s: %d", device_name.c_str(), err);
        }

        cl_mem_fill_fn =
            (cl_వరుసలో_చేర్చు_Mem_Fill_INTEL_fn) cl_Get_Extension_Function_చిరునామా_For_Platform(cl_platform, "clEnqueueMemFillINTEL");
        cl_mem_cpy_fn =
            (cl_వరుసలో_చేర్చు_Memcpy_INTEL_fn) cl_Get_Extension_Function_చిరునామా_For_Platform(cl_platform, "clEnqueueMemcpyINTEL");

        cl_ulong device_max_alloc = 0;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_MAX_MEM_ALLOC_కొలత, పరిమాణం(device_max_alloc), &device_max_alloc,
                              శూన్యసూచిక);
        ఒకవేళ (err == CL_SUCCESS) {
            max_alloc_కొలత = device_max_alloc;
        } లేకపోతే {
            // not fatal, ggml then allocates one buffer
            GGML_LOG_WARN("Failed to get OpenCL max allocation size: %d\n", err);
        }

        స్థిరరాశి cl_వరుస_properties profiling_properties[] = {
            CL_వరుస_PROPERTIES,
            CL_వరుస_PROFILING_ENABLE,
            0,
        };
        స్థిరరాశి cl_వరుస_properties * వరుస_properties =
            ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? profiling_properties : శూన్యసూచిక;
        cl_వరుస = cl_Create_Command_వరుస_With_Properties(cl_ctx, cl_device, వరుస_properties, &err);
        ఒకవేళ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to create the OpenCL queue for %s: %d", device_name.c_str(), err);
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_వరుస);
    } లేకపోతే ఒకవేళ (has_ఉపసర్గ(device_name, "NPU")) {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    ఒకవేళ (cl_వరుస != శూన్యసూచిక) {
        cl_Release_Command_వరుస(cl_వరుస);
        cl_వరుస = శూన్యసూచిక;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    స్థిర ggml_openvino_device_config config;
    తిరిగి config;
}

// Initialize device config (call during backend init)
శూన్య ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
స్థిరరాశి std::string & ggml_openvino_get_device_name() {
    తిరిగి ggml_openvino_get_device_config().device_name;
}

std::vector<std::string> ggml_openvino_get_అందుబాటు_devices() {
    స్వయంచాలక & config = ggml_openvino_get_device_config();
    config.init();
    తిరిగి config.అందుబాటు_devices;
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
స్థిరరాశి అక్షరం * ggml_openvino_getenv_str(స్థిరరాశి అక్షరం * var, స్థిరరాశి అక్షరం * default_విలువ) {
    స్వయంచాలక & env_map = ggml_openvino_get_device_config().environment_variables;
    స్వయంచాలక it = env_map.find(var);
    తిరిగి (it == env_map.end() || it->second.ఖాళీలు()) ? default_విలువ : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int (via std::atoi).
// Returns default_value (0) when the var is unset or empty. Used for both
// integer settings (e.g. GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean
// toggles: "0" disables, any non-zero integer enables.
పూర్ణాంకం ggml_openvino_getenv_int(స్థిరరాశి అక్షరం * var, పూర్ణాంకం default_విలువ) {
    స్థిరరాశి అక్షరం * v = ggml_openvino_getenv_str(var, శూన్యసూచిక);
    తిరిగి v ? std::atoi(v) : default_విలువ;
}

తార్కిక ggml_openvino_reduce_compile_mem_enabled() {
    స్థిరరాశి అక్షరం * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    ఒకవేళ (reduce_compile_mem != శూన్యసూచిక) {
        తిరిగి ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    తిరిగి ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

తార్కిక ggml_openvino_release_weights_enabled() {
    స్థిరరాశి అక్షరం * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    ఒకవేళ (release_weights != శూన్యసూచిక) {
        తిరిగి ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    తిరిగి ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
తార్కిక ggml_openvino_is_npu() {
    తిరిగి ggml_openvino_get_device_config().is_npu;
}

తార్కిక ggml_openvino_is_gpu() {
    తిరిగి has_ఉపసర్గ(ggml_openvino_get_device_name(), "GPU");
}

size_t ggml_openvino_max_alloc_కొలత() {
    తిరిగి ggml_openvino_get_device_config().max_alloc_కొలత;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    తిరిగి ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
స్థిరరాశి ov::AnyMap & ggml_openvino_get_compile_config() {
    తిరిగి ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_వరుస ggml_openvino_get_cl_వరుస() {
    తిరిగి ggml_openvino_get_device_config().cl_వరుస;
}

// Get the clEnqueueMemFillINTEL function pointer
cl_వరుసలో_చేర్చు_Mem_Fill_INTEL_fn ggml_openvino_get_cl_వరుసలో_చేర్చు_Mem_Fill_INTEL() {
    తిరిగి ggml_openvino_get_device_config().cl_mem_fill_fn;
}

// Get the clEnqueueMemcpyINTEL function pointer
cl_వరుసలో_చేర్చు_Memcpy_INTEL_fn ggml_openvino_get_cl_వరుసలో_చేర్చు_Memcpy_INTEL() {
    తిరిగి ggml_openvino_get_device_config().cl_mem_cpy_fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(స్థిరరాశి ggml_tensor * tensor, తార్కిక no_requant) {
    ఒకవేళ (no_requant) {
        తిరిగి std::nullopt;
    }
    ఒకవేళ (అక్షరశ్రేణి_పోలిక(tensor->name, "token_embd.weight", 17) == 0) {
        తిరిగి ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 :
                                                                             ExtraQuantType::Q8_0_C);
    }
    ఒకవేళ (అక్షరశ్రేణి_పోలిక(tensor->name, "output.weight", 13) == 0) {
        తిరిగి ExtraQuantType::Q8_0_C;
    }
    ఒకవేళ (ggml_openvino_is_npu()) {
        తిరిగి ExtraQuantType::Q4_0_128;
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
    స్థిరరాశి అక్షరం * rq = ggml_openvino_getenv_str("GGML_OPENVINO_REQUANT_KQUANT");
    స్వయంచాలక is_opt = [rq](స్థిరరాశి అక్షరం * name) {
        తిరిగి rq && strcmp(rq, name) == 0;
    };
    స్థిరరాశి తార్కిక sym128 = is_opt("q4_sym128");
    స్థిరరాశి తార్కిక sym128_all = is_opt("q4_sym128_all");
    స్థిరరాశి తార్కిక asym64_all = is_opt("q4_asym64_all");
    స్థిరరాశి తార్కిక asym64 = is_opt("q4_asym64");

    ఒకవేళ (tensor->type == GGML_TYPE_Q4_K) {
        ఒకవేళ (sym128_all) {
            తిరిగి ExtraQuantType::Q4_0_128;
        }
        ఒకవేళ (asym64_all) {
            తిరిగి ExtraQuantType::Q4_1_64;
        }
    }
    // MoE expert weights (3D, ne[2] = n_expert) stored as Q5_1/Q8_0 are the expert-side
    // equivalent of Q6_K/Q5_K: kept at 8 bits by default while the rest of the model is at 4
    // (gemma-4 26B-A4B keeps its down projection there). Send them to 4 bits under the same
    // option, at group 64 rather than 128: the down expert has k=704, which 64 divides
    // (704/64 = 11) and 128 does not.
    ఒకవేళ (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        ఒకవేళ (sym128 || sym128_all) {
            తిరిగి ExtraQuantType::Q4_0_64;
        }
        ఒకవేళ (asym64 || asym64_all) {
            తిరిగి ExtraQuantType::Q4_1_64;
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
        ఒకవేళ (ggml_openvino_is_gpu() && !is_opt("native")) {
            తిరిగి ExtraQuantType::Q4_0_64;
        }
    }
    ఎంపిక (tensor->type) {
    స్థితి GGML_TYPE_Q6_K:
    స్థితి GGML_TYPE_Q5_K:
        ఒకవేళ (sym128 || sym128_all) {
            తిరిగి ExtraQuantType::Q4_0_128;
        }
        ఒకవేళ (asym64 || asym64_all) {
            తిరిగి ExtraQuantType::Q4_1_64;
        }
        ఒకవేళ (is_opt("native")) {
            తిరిగి std::nullopt;
        }
        తిరిగి ExtraQuantType::Q8_0_C;
    అప్రమేయ:
        తిరిగి std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(స్థిరరాశి ggml_tensor * tensor, తార్కిక use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_సౌష్ఠవ = అబద్ధం;

    ఒకవేళ (!ggml_is_quantized(tensor->type)) {
        తిరిగి layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    ఒకవేళ (tensor->ne[3] != 1) {
        తిరిగి layout;
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    స్థిరరాశి తార్కిక for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    స్థిరరాశి size_t alignment = 64;  // Good for SIMD

    ఒకవేళ (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_చొప్పున_ఖండం = 32;
        layout.is_సౌష్ఠవ = నిజం;
        layout.weights_కొలత = ggml_nbytes(tensor);
        layout.weights_స్థానభ్రంశం = 0;
        layout.మొత్తం_కొలత = layout.weights_కొలత;
        తిరిగి layout;
    }

    // Check if requantization is needed (NPU-specific)
    స్వయంచాలక requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    ఒకవేళ (requant_type.has_value()) {
        layout.is_requant = నిజం;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        ఒకవేళ (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_కొలత = n_elements * పరిమాణం(uint16_t);  // F16 = 2 bytes
            layout.మొత్తం_కొలత = layout.weights_కొలత;
            layout.weights_స్థానభ్రంశం = 0;
            // No scales/zp for F16
            తిరిగి layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        ఎంపిక (requant_type.value()) {
        స్థితి ExtraQuantType::Q4_0_128:
            layout.is_u4 = నిజం;
            layout.weights_చొప్పున_ఖండం = 128;
            layout.is_సౌష్ఠవ = నిజం;
            ఆపు;
        స్థితి ExtraQuantType::Q4_1_64:
            layout.is_u4 = నిజం;
            layout.weights_చొప్పున_ఖండం = 64;
            layout.is_సౌష్ఠవ = అబద్ధం;
            ఆపు;
        స్థితి ExtraQuantType::Q4_0_64:
            layout.is_u4 = నిజం;
            layout.weights_చొప్పున_ఖండం = 64;
            layout.is_సౌష్ఠవ = నిజం;
            ఆపు;
        స్థితి ExtraQuantType::Q4_0_C:
            layout.is_u4 = నిజం;
            layout.weights_చొప్పున_ఖండం = tensor->ne[0];
            layout.is_సౌష్ఠవ = నిజం;
            ఆపు;
        స్థితి ExtraQuantType::Q8_0_32:
            layout.is_u4 = అబద్ధం;
            layout.weights_చొప్పున_ఖండం = 32;
            layout.is_సౌష్ఠవ = నిజం;
            ఆపు;
        స్థితి ExtraQuantType::Q8_0_C:
            layout.is_u4 = అబద్ధం;
            layout.weights_చొప్పున_ఖండం = tensor->ne[0];
            layout.is_సౌష్ఠవ = నిజం;
            ఆపు;
        స్థితి ExtraQuantType::Q8_1_C:
            layout.is_u4 = అబద్ధం;
            layout.weights_చొప్పున_ఖండం = tensor->ne[0];
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }

        ఒకవేళ (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_కొలత = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_ఖండాలు = n_elements / layout.weights_చొప్పున_ఖండం;
            layout.scales_కొలత = n_ఖండాలు * పరిమాణం(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            ఒకవేళ (layout.is_సౌష్ఠవ) {
                layout.zp_కొలత = 0;
            } లేకపోతే {
                layout.zp_కొలత = layout.is_u4 ? ((n_ఖండాలు + 1) / 2) : n_ఖండాలు;
            }

            layout.weights_స్థానభ్రంశం = 0;
            layout.scales_స్థానభ్రంశం = ((layout.weights_కొలత + alignment - 1) / alignment) * alignment;
            layout.zp_స్థానభ్రంశం = layout.scales_స్థానభ్రంశం + ((layout.scales_కొలత + alignment - 1) / alignment) * alignment;
            layout.మొత్తం_కొలత = layout.zp_స్థానభ్రంశం + layout.zp_కొలత;
            layout.మొత్తం_కొలత = std::max(layout.మొత్తం_కొలత, ggml_nbytes(tensor));
            తిరిగి layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = అబద్ధం;
    layout.weights_చొప్పున_ఖండం = 32;
    layout.is_సౌష్ఠవ = అబద్ధం;

    ఎంపిక (tensor->type) {
    స్థితి GGML_TYPE_MXFP4:
    స్థితి GGML_TYPE_Q4_0:
        layout.is_u4 = నిజం;
        layout.is_సౌష్ఠవ = నిజం;
        ఆపు;

    స్థితి GGML_TYPE_Q4_1:
    స్థితి GGML_TYPE_Q4_K:
        layout.is_u4 = నిజం;
        ఆపు;

    స్థితి GGML_TYPE_Q8_0:
        layout.is_సౌష్ఠవ = నిజం;
        ఆపు;

    స్థితి GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        ఆపు;

    స్థితి GGML_TYPE_Q6_K:
        layout.weights_చొప్పున_ఖండం = 16;
        layout.is_సౌష్ఠవ = నిజం;
        ఆపు;

    స్థితి GGML_TYPE_Q5_K:
        ఆపు;

    అప్రమేయ:
        // Unsupported quantization type
        తిరిగి layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_కొలత = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_ఖండాలు = n_elements / layout.weights_చొప్పున_ఖండం;
    layout.scales_కొలత = n_ఖండాలు * (tensor->type == GGML_TYPE_MXFP4 ? పరిమాణం(uint8_t) : పరిమాణం(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    ఒకవేళ (layout.is_సౌష్ఠవ) {
        layout.zp_కొలత = 0;
    } లేకపోతే ఒకవేళ (use_bias || for_gather_matmul) {
        layout.zp_కొలత = n_ఖండాలు * పరిమాణం(uint16_t);
    } లేకపోతే {
        layout.zp_కొలత = layout.is_u4 ? ((n_ఖండాలు + 1) / 2) : n_ఖండాలు;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_స్థానభ్రంశం = 0;
    layout.scales_స్థానభ్రంశం = ((layout.weights_కొలత + alignment - 1) / alignment) * alignment;
    layout.zp_స్థానభ్రంశం = layout.scales_స్థానభ్రంశం + ((layout.scales_కొలత + alignment - 1) / alignment) * alignment;
    layout.మొత్తం_కొలత = layout.zp_స్థానభ్రంశం + layout.zp_కొలత;
    layout.మొత్తం_కొలత = std::max(layout.మొత్తం_కొలత, ggml_nbytes(tensor));

    తిరిగి layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(స్థిరరాశి ggml_tensor * tensor, తార్కిక is_remote) {
    ov::Shape shape;
    కొరకు (పూర్ణాంకం i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    ఎంపిక (tensor->type) {
    స్థితి GGML_TYPE_F32:
        element_type = ov::element::f32;
        ఆపు;
    స్థితి GGML_TYPE_F16:
        element_type = ov::element::f16;
        ఆపు;
    స్థితి GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        ఆపు;
    స్థితి GGML_TYPE_I32:
        element_type = ov::element::i32;
        ఆపు;
    స్థితి GGML_TYPE_I64:
        element_type = ov::element::i64;
        ఆపు;
    అప్రమేయ:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        తిరిగి శూన్యసూచిక;
    }

    స్వయంచాలక remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    ఒకవేళ (is_remote) {
        GGML_ASSERT(ggml_openvino_is_gpu());
        స్వయంచాలక gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        స్వయంచాలక usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->దత్తాంశం);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } లేకపోతే {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->దత్తాంశం);
    }

    తిరిగి కొత్త ggml_openvino_tensor_extra(ov_tensor);
}
