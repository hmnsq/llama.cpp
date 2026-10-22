#அந்தர்பாவ "ggml-openvino-extra.h"

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "model-cache.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#அந்தர்பாவ <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#அந்தர்பாவ <openvino/runtime/properties.hpp>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <optional>

ov::Core & ov_singleton_core() {
    நிலையான ov::Core core;
    திரும்பு core;
}

நிலையான தருக்க has_prefix(மாறிலி std::string & s, மாறிலி std::string & prefix) {
    திரும்பு s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

நிலையான தருக்க is_virtual_routing_device(மாறிலி std::string & device_name) {
    திரும்பு has_prefix(device_name, "AUTO") || has_prefix(device_name, "MULTI") || has_prefix(device_name, "HETERO");
}

நிலையான std::vector<std::string> ov_enumerate_devices() {
    std::vector<std::string> result;

    க்கு (மாறிலி auto & device : ov_singleton_core().get_available_devices()) {
        எனில் (!is_virtual_routing_device(device)) {
            result.push_back(device);
        }
    }

    எனில் (result.empty()) {
        result.push_back("CPU");
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    திரும்பு result;
}

std::string ggml_openvino_get_device_description(மாறிலி std::string & device_name) {
    std::string description = device_name;
    முயல் {
        description = ov_singleton_core().get_property(device_name, ov::device::full_name);
    } பிடி (...) {
        திரும்பு device_name;
    }

    எனில் (has_prefix(device_name, "NPU")) {
        முயல் {
            மாறிலி std::string arch = ov_singleton_core().get_property(device_name, "DEVICE_ARCHITECTURE").as<std::string>();
            எனில் (!arch.empty()) {
                description += " (NPU " + arch + ")";
            }
        } பிடி (...) {
        }
    }

    திரும்பு description;
}

// requested: GGML_OPENVINO_DEVICE, nullptr if unset. available_devices is never empty (see ov_enumerate_devices)
நிலையான std::string resolve_openvino_device_name(மாறிலி std::vector<std::string> & available_devices,
                                                மாறிலி எழுத்து * requested) {
    auto available = [&](மாறிலி std::string & name) {
        திரும்பு std::find(available_devices.begin(), available_devices.end(), name) != available_devices.end();
    };
    எனில் (requested == nullptr) {
        திரும்பு available("CPU") ? "CPU" : available_devices.front();
    }
    எனில் (!available(requested)) {
        // No fallback to CPU (easy to miss) and no GPU -> GPU.0 alias (with iGPU + dGPU, GPU.0 is often the
        // wrong one). List the devices here: --list-devices initializes this backend and would abort too.
        std::string list;
        க்கு (மாறிலி std::string & name : available_devices) {
            list += "\n  " + name + ": " + ggml_openvino_get_device_description(name);
        }
        GGML_ABORT("GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s is not available. "
                   "Set it to one of the available OpenVINO devices:%s",
                   requested, list.c_str());
    }
    திரும்பு requested;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

வெறுமை ggml_openvino_device_config::init() {
    நிலையான std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    எனில் (initialized) {
        திரும்பு;
    }
    // Set up front: a failed OpenCL setup below is not retried on every call
    initialized = உண்மை;

    // All recognized GGML_OPENVINO_* env vars. Their values are cached here
    // once at backend init time and read back via ggml_openvino_getenv_str()
    // (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
    நிலையான constexpr மாறிலி எழுத்து * env_var_names[] = {
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

    க்கு (மாறிலி எழுத்து * மாறிலி & env_var : env_var_names) {
        auto * env = getenv(env_var);
        எனில் (env) {
            environment_variables[env_var] = env;
        }
    }

    available_devices = ov_enumerate_devices();
    device_name = resolve_openvino_device_name(available_devices, ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE"));
    is_npu = has_prefix(device_name, "NPU");

    ggml_openvino_model_cache_init();

    மாறிலி எழுத்து * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    எனில் (has_prefix(device_name, "NPU")) {
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
        எனில் (cache_dir && strlen(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        மாறிலி எழுத்து * compilation_mode_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        எனில் (compilation_mode_params && strlen(compilation_mode_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_mode_params;
        }
    } இல்லையெனில் எனில் (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }

    எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2) {
        compile_config.insert(ov::enable_profiling(உண்மை));
    }

    // Initialize remote context with queue sharing for GPU
    எனில் (has_prefix(device_name, "GPU")) {
        // Use the OpenCL context OpenVINO created for this device, so GPU.N gets its own device
        cl_context cl_ctx;
        முயல் {
            auto ov_ctx = ov_singleton_core().get_default_context(device_name).as<ov::intel_gpu::ocl::ClContext>();
            cl_ctx = ov_ctx.get();
        } பிடி (மாறிலி std::exception & e) {
            // The consumers of the remote context have no host fallback, and OpenVINO
            // already reported the device as present.
            GGML_ABORT("ggml-openvino: failed to get the OpenCL context for %s: %s", device_name.c_str(), e.what());
        }

        cl_int err;
        cl_device_id cl_device;
        err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, அளவு(cl_device), &cl_device, nullptr);
        எனில் (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL device for %s: %d", device_name.c_str(), err);
        }

        cl_platform_id cl_platform;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_PLATFORM, அளவு(cl_platform), &cl_platform, nullptr);
        எனில் (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL platform for %s: %d", device_name.c_str(), err);
        }

        cl_mem_fill_fn =
            (clEnqueueMemFillINTEL_fn) clGetExtensionFunctionAddressForPlatform(cl_platform, "clEnqueueMemFillINTEL");
        cl_mem_cpy_fn =
            (clEnqueueMemcpyINTEL_fn) clGetExtensionFunctionAddressForPlatform(cl_platform, "clEnqueueMemcpyINTEL");

        cl_ulong device_max_alloc = 0;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, அளவு(device_max_alloc), &device_max_alloc,
                              nullptr);
        எனில் (err == CL_SUCCESS) {
            max_alloc_size = device_max_alloc;
        } இல்லையெனில் {
            // not fatal, ggml then allocates one buffer
            GGML_LOG_WARN("Failed to get OpenCL max allocation size: %d\n", err);
        }

        மாறிலி cl_queue_properties profiling_properties[] = {
            CL_QUEUE_PROPERTIES,
            CL_QUEUE_PROFILING_ENABLE,
            0,
        };
        மாறிலி cl_queue_properties * queue_properties =
            ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? profiling_properties : nullptr;
        cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, queue_properties, &err);
        எனில் (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to create the OpenCL queue for %s: %d", device_name.c_str(), err);
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_queue);
    } இல்லையெனில் எனில் (has_prefix(device_name, "NPU")) {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    எனில் (cl_queue != nullptr) {
        clReleaseCommandQueue(cl_queue);
        cl_queue = nullptr;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    நிலையான ggml_openvino_device_config config;
    திரும்பு config;
}

// Initialize device config (call during backend init)
வெறுமை ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
மாறிலி std::string & ggml_openvino_get_device_name() {
    திரும்பு ggml_openvino_get_device_config().device_name;
}

std::vector<std::string> ggml_openvino_get_available_devices() {
    auto & config = ggml_openvino_get_device_config();
    config.init();
    திரும்பு config.available_devices;
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
மாறிலி எழுத்து * ggml_openvino_getenv_str(மாறிலி எழுத்து * var, மாறிலி எழுத்து * default_மதிப்பு) {
    auto & env_map = ggml_openvino_get_device_config().environment_variables;
    auto it = env_map.find(var);
    திரும்பு (it == env_map.end() || it->second.empty()) ? default_மதிப்பு : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int (via std::atoi).
// Returns default_value (0) when the var is unset or empty. Used for both
// integer settings (e.g. GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean
// toggles: "0" disables, any non-zero integer enables.
முழு ggml_openvino_getenv_int(மாறிலி எழுத்து * var, முழு default_மதிப்பு) {
    மாறிலி எழுத்து * v = ggml_openvino_getenv_str(var, nullptr);
    திரும்பு v ? std::atoi(v) : default_மதிப்பு;
}

தருக்க ggml_openvino_reduce_compile_mem_enabled() {
    மாறிலி எழுத்து * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    எனில் (reduce_compile_mem != nullptr) {
        திரும்பு ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    திரும்பு ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

தருக்க ggml_openvino_release_weights_enabled() {
    மாறிலி எழுத்து * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    எனில் (release_weights != nullptr) {
        திரும்பு ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    திரும்பு ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
தருக்க ggml_openvino_is_npu() {
    திரும்பு ggml_openvino_get_device_config().is_npu;
}

தருக்க ggml_openvino_is_gpu() {
    திரும்பு has_prefix(ggml_openvino_get_device_name(), "GPU");
}

size_t ggml_openvino_max_alloc_size() {
    திரும்பு ggml_openvino_get_device_config().max_alloc_size;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    திரும்பு ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
மாறிலி ov::AnyMap & ggml_openvino_get_compile_config() {
    திரும்பு ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_queue ggml_openvino_get_cl_queue() {
    திரும்பு ggml_openvino_get_device_config().cl_queue;
}

// Get the clEnqueueMemFillINTEL function pointer
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL() {
    திரும்பு ggml_openvino_get_device_config().cl_mem_fill_fn;
}

// Get the clEnqueueMemcpyINTEL function pointer
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL() {
    திரும்பு ggml_openvino_get_device_config().cl_mem_cpy_fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(மாறிலி ggml_tensor * tensor, தருக்க no_requant) {
    எனில் (no_requant) {
        திரும்பு std::nullopt;
    }
    எனில் (strncmp(tensor->name, "token_embd.weight", 17) == 0) {
        திரும்பு ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 :
                                                                             ExtraQuantType::Q8_0_C);
    }
    எனில் (strncmp(tensor->name, "output.weight", 13) == 0) {
        திரும்பு ExtraQuantType::Q8_0_C;
    }
    எனில் (ggml_openvino_is_npu()) {
        திரும்பு ExtraQuantType::Q4_0_128;
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
    மாறிலி எழுத்து * rq = ggml_openvino_getenv_str("GGML_OPENVINO_REQUANT_KQUANT");
    auto is_opt = [rq](மாறிலி எழுத்து * name) {
        திரும்பு rq && strcmp(rq, name) == 0;
    };
    மாறிலி தருக்க sym128 = is_opt("q4_sym128");
    மாறிலி தருக்க sym128_all = is_opt("q4_sym128_all");
    மாறிலி தருக்க asym64_all = is_opt("q4_asym64_all");
    மாறிலி தருக்க asym64 = is_opt("q4_asym64");

    எனில் (tensor->type == GGML_TYPE_Q4_K) {
        எனில் (sym128_all) {
            திரும்பு ExtraQuantType::Q4_0_128;
        }
        எனில் (asym64_all) {
            திரும்பு ExtraQuantType::Q4_1_64;
        }
    }
    // MoE expert weights (3D, ne[2] = n_expert) stored as Q5_1/Q8_0 are the expert-side
    // equivalent of Q6_K/Q5_K: kept at 8 bits by default while the rest of the model is at 4
    // (gemma-4 26B-A4B keeps its down projection there). Send them to 4 bits under the same
    // option, at group 64 rather than 128: the down expert has k=704, which 64 divides
    // (704/64 = 11) and 128 does not.
    எனில் (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        எனில் (sym128 || sym128_all) {
            திரும்பு ExtraQuantType::Q4_0_64;
        }
        எனில் (asym64 || asym64_all) {
            திரும்பு ExtraQuantType::Q4_1_64;
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
        எனில் (ggml_openvino_is_gpu() && !is_opt("native")) {
            திரும்பு ExtraQuantType::Q4_0_64;
        }
    }
    தேர்வு (tensor->type) {
    நிலை GGML_TYPE_Q6_K:
    நிலை GGML_TYPE_Q5_K:
        எனில் (sym128 || sym128_all) {
            திரும்பு ExtraQuantType::Q4_0_128;
        }
        எனில் (asym64 || asym64_all) {
            திரும்பு ExtraQuantType::Q4_1_64;
        }
        எனில் (is_opt("native")) {
            திரும்பு std::nullopt;
        }
        திரும்பு ExtraQuantType::Q8_0_C;
    இயல்பு:
        திரும்பு std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(மாறிலி ggml_tensor * tensor, தருக்க use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_symmetric = பொய்;

    எனில் (!ggml_is_quantized(tensor->type)) {
        திரும்பு layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    எனில் (tensor->ne[3] != 1) {
        திரும்பு layout;
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    மாறிலி தருக்க for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    மாறிலி size_t alignment = 64;  // Good for SIMD

    எனில் (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_per_block = 32;
        layout.is_symmetric = உண்மை;
        layout.weights_size = ggml_nbytes(tensor);
        layout.weights_offset = 0;
        layout.total_size = layout.weights_size;
        திரும்பு layout;
    }

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    எனில் (requant_type.has_value()) {
        layout.is_requant = உண்மை;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        எனில் (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_size = n_elements * அளவு(uint16_t);  // F16 = 2 bytes
            layout.total_size = layout.weights_size;
            layout.weights_offset = 0;
            // No scales/zp for F16
            திரும்பு layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        தேர்வு (requant_type.value()) {
        நிலை ExtraQuantType::Q4_0_128:
            layout.is_u4 = உண்மை;
            layout.weights_per_block = 128;
            layout.is_symmetric = உண்மை;
            நிறுத்து;
        நிலை ExtraQuantType::Q4_1_64:
            layout.is_u4 = உண்மை;
            layout.weights_per_block = 64;
            layout.is_symmetric = பொய்;
            நிறுத்து;
        நிலை ExtraQuantType::Q4_0_64:
            layout.is_u4 = உண்மை;
            layout.weights_per_block = 64;
            layout.is_symmetric = உண்மை;
            நிறுத்து;
        நிலை ExtraQuantType::Q4_0_C:
            layout.is_u4 = உண்மை;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = உண்மை;
            நிறுத்து;
        நிலை ExtraQuantType::Q8_0_32:
            layout.is_u4 = பொய்;
            layout.weights_per_block = 32;
            layout.is_symmetric = உண்மை;
            நிறுத்து;
        நிலை ExtraQuantType::Q8_0_C:
            layout.is_u4 = பொய்;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = உண்மை;
            நிறுத்து;
        நிலை ExtraQuantType::Q8_1_C:
            layout.is_u4 = பொய்;
            layout.weights_per_block = tensor->ne[0];
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }

        எனில் (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / layout.weights_per_block;
            layout.scales_size = n_blocks * அளவு(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            எனில் (layout.is_symmetric) {
                layout.zp_size = 0;
            } இல்லையெனில் {
                layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
            }

            layout.weights_offset = 0;
            layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
            layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
            layout.total_size = layout.zp_offset + layout.zp_size;
            layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));
            திரும்பு layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = பொய்;
    layout.weights_per_block = 32;
    layout.is_symmetric = பொய்;

    தேர்வு (tensor->type) {
    நிலை GGML_TYPE_MXFP4:
    நிலை GGML_TYPE_Q4_0:
        layout.is_u4 = உண்மை;
        layout.is_symmetric = உண்மை;
        நிறுத்து;

    நிலை GGML_TYPE_Q4_1:
    நிலை GGML_TYPE_Q4_K:
        layout.is_u4 = உண்மை;
        நிறுத்து;

    நிலை GGML_TYPE_Q8_0:
        layout.is_symmetric = உண்மை;
        நிறுத்து;

    நிலை GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        நிறுத்து;

    நிலை GGML_TYPE_Q6_K:
        layout.weights_per_block = 16;
        layout.is_symmetric = உண்மை;
        நிறுத்து;

    நிலை GGML_TYPE_Q5_K:
        நிறுத்து;

    இயல்பு:
        // Unsupported quantization type
        திரும்பு layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_blocks = n_elements / layout.weights_per_block;
    layout.scales_size = n_blocks * (tensor->type == GGML_TYPE_MXFP4 ? அளவு(uint8_t) : அளவு(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    எனில் (layout.is_symmetric) {
        layout.zp_size = 0;
    } இல்லையெனில் எனில் (use_bias || for_gather_matmul) {
        layout.zp_size = n_blocks * அளவு(uint16_t);
    } இல்லையெனில் {
        layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_offset = 0;
    layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
    layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
    layout.total_size = layout.zp_offset + layout.zp_size;
    layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));

    திரும்பு layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(மாறிலி ggml_tensor * tensor, தருக்க is_remote) {
    ov::Shape shape;
    க்கு (முழு i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    தேர்வு (tensor->type) {
    நிலை GGML_TYPE_F32:
        element_type = ov::element::f32;
        நிறுத்து;
    நிலை GGML_TYPE_F16:
        element_type = ov::element::f16;
        நிறுத்து;
    நிலை GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        நிறுத்து;
    நிலை GGML_TYPE_I32:
        element_type = ov::element::i32;
        நிறுத்து;
    நிலை GGML_TYPE_I64:
        element_type = ov::element::i64;
        நிறுத்து;
    இயல்பு:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        திரும்பு nullptr;
    }

    auto remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    எனில் (is_remote) {
        GGML_ASSERT(ggml_openvino_is_gpu());
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        auto usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->data);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } இல்லையெனில் {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->data);
    }

    திரும்பு புதிய ggml_openvino_tensor_extra(ov_tensor);
}
