#ضمن "ggml-openvino-extra.h"

#ضمن "ggml-impl.h"
#ضمن "ggml.h"
#ضمن "model-cache.h"

#ضمن <algorithm>
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#ضمن <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#ضمن <openvino/runtime/properties.hpp>
#ضمن <mutex>
#ضمن <optional>

ov::Core & ov_singleton_core() {
    ساكن ov::Core core;
    ܦܢܐ core;
}

ساكن منطقي has_prefix(ثابت std::string & s, ثابت std::string & prefix) {
    ܦܢܐ s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

ساكن منطقي is_virtual_routing_device(ثابت std::string & device_name) {
    ܦܢܐ has_prefix(device_name, "AUTO") || has_prefix(device_name, "MULTI") || has_prefix(device_name, "HETERO");
}

ساكن std::vector<std::string> ov_enumerate_devices() {
    std::vector<std::string> result;

    لكل (ثابت auto & device : ov_singleton_core().get_available_devices()) {
        ܐܢ (!is_virtual_routing_device(device)) {
            result.push_back(device);
        }
    }

    ܐܢ (result.empty()) {
        result.push_back("CPU");
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    ܦܢܐ result;
}

std::string ggml_openvino_get_device_description(ثابت std::string & device_name) {
    std::string description = device_name;
    حاول {
        description = ov_singleton_core().get_property(device_name, ov::device::full_name);
    } امسك (...) {
        ܦܢܐ device_name;
    }

    ܐܢ (has_prefix(device_name, "NPU")) {
        حاول {
            ثابت std::string arch = ov_singleton_core().get_property(device_name, "DEVICE_ARCHITECTURE").as<std::string>();
            ܐܢ (!arch.empty()) {
                description += " (NPU " + arch + ")";
            }
        } امسك (...) {
        }
    }

    ܦܢܐ description;
}

// requested: GGML_OPENVINO_DEVICE, nullptr if unset. available_devices is never empty (see ov_enumerate_devices)
ساكن std::string resolve_openvino_device_name(ثابت std::vector<std::string> & available_devices,
                                                ثابت حرف * requested) {
    auto available = [&](ثابت std::string & name) {
        ܦܢܐ std::find(available_devices.begin(), available_devices.end(), name) != available_devices.end();
    };
    ܐܢ (requested == nullptr) {
        ܦܢܐ available("CPU") ? "CPU" : available_devices.front();
    }
    ܐܢ (!available(requested)) {
        // No fallback to CPU (easy to miss) and no GPU -> GPU.0 alias (with iGPU + dGPU, GPU.0 is often the
        // wrong one). List the devices here: --list-devices initializes this backend and would abort too.
        std::string list;
        لكل (ثابت std::string & name : available_devices) {
            list += "\n  " + name + ": " + ggml_openvino_get_device_description(name);
        }
        GGML_ABORT("GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s is not available. "
                   "Set it to one of the available OpenVINO devices:%s",
                   requested, list.c_str());
    }
    ܦܢܐ requested;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

فراغ ggml_openvino_device_config::init() {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    ܐܢ (initialized) {
        ܦܢܐ;
    }
    // Set up front: a failed OpenCL setup below is not retried on every call
    initialized = ܫܪܪܐ;

    // All recognized GGML_OPENVINO_* env vars. Their values are cached here
    // once at backend init time and read back via ggml_openvino_getenv_str()
    // (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
    ساكن constexpr ثابت حرف * env_var_names[] = {
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

    لكل (ثابت حرف * ثابت & env_var : env_var_names) {
        auto * env = getenv(env_var);
        ܐܢ (env) {
            environment_variables[env_var] = env;
        }
    }

    available_devices = ov_enumerate_devices();
    device_name = resolve_openvino_device_name(available_devices, ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE"));
    is_npu = has_prefix(device_name, "NPU");

    ggml_openvino_model_cache_init();

    ثابت حرف * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    ܐܢ (has_prefix(device_name, "NPU")) {
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
        ܐܢ (cache_dir && strlen(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        ثابت حرف * compilation_mode_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        ܐܢ (compilation_mode_params && strlen(compilation_mode_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_mode_params;
        }
    } ܐܚܪܢܐ ܐܢ (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }

    ܐܢ (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2) {
        compile_config.insert(ov::enable_profiling(ܫܪܪܐ));
    }

    // Initialize remote context with queue sharing for GPU
    ܐܢ (has_prefix(device_name, "GPU")) {
        // Use the OpenCL context OpenVINO created for this device, so GPU.N gets its own device
        cl_context cl_ctx;
        حاول {
            auto ov_ctx = ov_singleton_core().get_default_context(device_name).as<ov::intel_gpu::ocl::ClContext>();
            cl_ctx = ov_ctx.get();
        } امسك (ثابت std::exception & e) {
            // The consumers of the remote context have no host fallback, and OpenVINO
            // already reported the device as present.
            GGML_ABORT("ggml-openvino: failed to get the OpenCL context for %s: %s", device_name.c_str(), e.what());
        }

        cl_int err;
        cl_device_id cl_device;
        err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, حجم(cl_device), &cl_device, nullptr);
        ܐܢ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL device for %s: %d", device_name.c_str(), err);
        }

        cl_platform_id cl_platform;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_PLATFORM, حجم(cl_platform), &cl_platform, nullptr);
        ܐܢ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL platform for %s: %d", device_name.c_str(), err);
        }

        cl_mem_fill_fn =
            (clEnqueueMemFillINTEL_fn) clGetExtensionFunctionAddressForPlatform(cl_platform, "clEnqueueMemFillINTEL");
        cl_mem_cpy_fn =
            (clEnqueueMemcpyINTEL_fn) clGetExtensionFunctionAddressForPlatform(cl_platform, "clEnqueueMemcpyINTEL");

        cl_ulong device_max_alloc = 0;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, حجم(device_max_alloc), &device_max_alloc,
                              nullptr);
        ܐܢ (err == CL_SUCCESS) {
            max_alloc_size = device_max_alloc;
        } ܐܚܪܢܐ {
            // not fatal, ggml then allocates one buffer
            GGML_LOG_WARN("Failed to get OpenCL max allocation size: %d\n", err);
        }

        ثابت cl_queue_properties profiling_properties[] = {
            CL_QUEUE_PROPERTIES,
            CL_QUEUE_PROFILING_ENABLE,
            0,
        };
        ثابت cl_queue_properties * queue_properties =
            ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? profiling_properties : nullptr;
        cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, queue_properties, &err);
        ܐܢ (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to create the OpenCL queue for %s: %d", device_name.c_str(), err);
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_queue);
    } ܐܚܪܢܐ ܐܢ (has_prefix(device_name, "NPU")) {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    ܐܢ (cl_queue != nullptr) {
        clReleaseCommandQueue(cl_queue);
        cl_queue = nullptr;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    ساكن ggml_openvino_device_config config;
    ܦܢܐ config;
}

// Initialize device config (call during backend init)
فراغ ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
ثابت std::string & ggml_openvino_get_device_name() {
    ܦܢܐ ggml_openvino_get_device_config().device_name;
}

std::vector<std::string> ggml_openvino_get_available_devices() {
    auto & config = ggml_openvino_get_device_config();
    config.init();
    ܦܢܐ config.available_devices;
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
ثابت حرف * ggml_openvino_getenv_str(ثابت حرف * var, ثابت حرف * default_value) {
    auto & env_map = ggml_openvino_get_device_config().environment_variables;
    auto it = env_map.find(var);
    ܦܢܐ (it == env_map.end() || it->second.empty()) ? default_value : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int (via std::atoi).
// Returns default_value (0) when the var is unset or empty. Used for both
// integer settings (e.g. GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean
// toggles: "0" disables, any non-zero integer enables.
صحيح ggml_openvino_getenv_int(ثابت حرف * var, صحيح default_value) {
    ثابت حرف * v = ggml_openvino_getenv_str(var, nullptr);
    ܦܢܐ v ? std::atoi(v) : default_value;
}

منطقي ggml_openvino_reduce_compile_mem_enabled() {
    ثابت حرف * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    ܐܢ (reduce_compile_mem != nullptr) {
        ܦܢܐ ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    ܦܢܐ ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

منطقي ggml_openvino_release_weights_enabled() {
    ثابت حرف * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    ܐܢ (release_weights != nullptr) {
        ܦܢܐ ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    ܦܢܐ ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
منطقي ggml_openvino_is_npu() {
    ܦܢܐ ggml_openvino_get_device_config().is_npu;
}

منطقي ggml_openvino_is_gpu() {
    ܦܢܐ has_prefix(ggml_openvino_get_device_name(), "GPU");
}

size_t ggml_openvino_max_alloc_size() {
    ܦܢܐ ggml_openvino_get_device_config().max_alloc_size;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    ܦܢܐ ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
ثابت ov::AnyMap & ggml_openvino_get_compile_config() {
    ܦܢܐ ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_queue ggml_openvino_get_cl_queue() {
    ܦܢܐ ggml_openvino_get_device_config().cl_queue;
}

// Get the clEnqueueMemFillINTEL function pointer
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL() {
    ܦܢܐ ggml_openvino_get_device_config().cl_mem_fill_fn;
}

// Get the clEnqueueMemcpyINTEL function pointer
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL() {
    ܦܢܐ ggml_openvino_get_device_config().cl_mem_cpy_fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ثابت ggml_tensor * tensor, منطقي no_requant) {
    ܐܢ (no_requant) {
        ܦܢܐ std::nullopt;
    }
    ܐܢ (strncmp(tensor->name, "token_embd.weight", 17) == 0) {
        ܦܢܐ ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 :
                                                                             ExtraQuantType::Q8_0_C);
    }
    ܐܢ (strncmp(tensor->name, "output.weight", 13) == 0) {
        ܦܢܐ ExtraQuantType::Q8_0_C;
    }
    ܐܢ (ggml_openvino_is_npu()) {
        ܦܢܐ ExtraQuantType::Q4_0_128;
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
    ثابت حرف * rq = ggml_openvino_getenv_str("GGML_OPENVINO_REQUANT_KQUANT");
    auto is_opt = [rq](ثابت حرف * name) {
        ܦܢܐ rq && strcmp(rq, name) == 0;
    };
    ثابت منطقي sym128 = is_opt("q4_sym128");
    ثابت منطقي sym128_all = is_opt("q4_sym128_all");
    ثابت منطقي asym64_all = is_opt("q4_asym64_all");
    ثابت منطقي asym64 = is_opt("q4_asym64");

    ܐܢ (tensor->type == GGML_TYPE_Q4_K) {
        ܐܢ (sym128_all) {
            ܦܢܐ ExtraQuantType::Q4_0_128;
        }
        ܐܢ (asym64_all) {
            ܦܢܐ ExtraQuantType::Q4_1_64;
        }
    }
    // MoE expert weights (3D, ne[2] = n_expert) stored as Q5_1/Q8_0 are the expert-side
    // equivalent of Q6_K/Q5_K: kept at 8 bits by default while the rest of the model is at 4
    // (gemma-4 26B-A4B keeps its down projection there). Send them to 4 bits under the same
    // option, at group 64 rather than 128: the down expert has k=704, which 64 divides
    // (704/64 = 11) and 128 does not.
    ܐܢ (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        ܐܢ (sym128 || sym128_all) {
            ܦܢܐ ExtraQuantType::Q4_0_64;
        }
        ܐܢ (asym64 || asym64_all) {
            ܦܢܐ ExtraQuantType::Q4_1_64;
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
        ܐܢ (ggml_openvino_is_gpu() && !is_opt("native")) {
            ܦܢܐ ExtraQuantType::Q4_0_64;
        }
    }
    بدل (tensor->type) {
    حالة GGML_TYPE_Q6_K:
    حالة GGML_TYPE_Q5_K:
        ܐܢ (sym128 || sym128_all) {
            ܦܢܐ ExtraQuantType::Q4_0_128;
        }
        ܐܢ (asym64 || asym64_all) {
            ܦܢܐ ExtraQuantType::Q4_1_64;
        }
        ܐܢ (is_opt("native")) {
            ܦܢܐ std::nullopt;
        }
        ܦܢܐ ExtraQuantType::Q8_0_C;
    افتراضي:
        ܦܢܐ std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(ثابت ggml_tensor * tensor, منطقي use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_symmetric = ܕܓܠܐ;

    ܐܢ (!ggml_is_quantized(tensor->type)) {
        ܦܢܐ layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    ܐܢ (tensor->ne[3] != 1) {
        ܦܢܐ layout;
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    ثابت منطقي for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    ثابت size_t alignment = 64;  // Good for SIMD

    ܐܢ (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_per_block = 32;
        layout.is_symmetric = ܫܪܪܐ;
        layout.weights_size = ggml_nbytes(tensor);
        layout.weights_offset = 0;
        layout.total_size = layout.weights_size;
        ܦܢܐ layout;
    }

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    ܐܢ (requant_type.has_value()) {
        layout.is_requant = ܫܪܪܐ;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        ܐܢ (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_size = n_elements * حجم(uint16_t);  // F16 = 2 bytes
            layout.total_size = layout.weights_size;
            layout.weights_offset = 0;
            // No scales/zp for F16
            ܦܢܐ layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        بدل (requant_type.value()) {
        حالة ExtraQuantType::Q4_0_128:
            layout.is_u4 = ܫܪܪܐ;
            layout.weights_per_block = 128;
            layout.is_symmetric = ܫܪܪܐ;
            اكسر;
        حالة ExtraQuantType::Q4_1_64:
            layout.is_u4 = ܫܪܪܐ;
            layout.weights_per_block = 64;
            layout.is_symmetric = ܕܓܠܐ;
            اكسر;
        حالة ExtraQuantType::Q4_0_64:
            layout.is_u4 = ܫܪܪܐ;
            layout.weights_per_block = 64;
            layout.is_symmetric = ܫܪܪܐ;
            اكسر;
        حالة ExtraQuantType::Q4_0_C:
            layout.is_u4 = ܫܪܪܐ;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = ܫܪܪܐ;
            اكسر;
        حالة ExtraQuantType::Q8_0_32:
            layout.is_u4 = ܕܓܠܐ;
            layout.weights_per_block = 32;
            layout.is_symmetric = ܫܪܪܐ;
            اكسر;
        حالة ExtraQuantType::Q8_0_C:
            layout.is_u4 = ܕܓܠܐ;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = ܫܪܪܐ;
            اكسر;
        حالة ExtraQuantType::Q8_1_C:
            layout.is_u4 = ܕܓܠܐ;
            layout.weights_per_block = tensor->ne[0];
            اكسر;
        افتراضي:
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }

        ܐܢ (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / layout.weights_per_block;
            layout.scales_size = n_blocks * حجم(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            ܐܢ (layout.is_symmetric) {
                layout.zp_size = 0;
            } ܐܚܪܢܐ {
                layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
            }

            layout.weights_offset = 0;
            layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
            layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
            layout.total_size = layout.zp_offset + layout.zp_size;
            layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));
            ܦܢܐ layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = ܕܓܠܐ;
    layout.weights_per_block = 32;
    layout.is_symmetric = ܕܓܠܐ;

    بدل (tensor->type) {
    حالة GGML_TYPE_MXFP4:
    حالة GGML_TYPE_Q4_0:
        layout.is_u4 = ܫܪܪܐ;
        layout.is_symmetric = ܫܪܪܐ;
        اكسر;

    حالة GGML_TYPE_Q4_1:
    حالة GGML_TYPE_Q4_K:
        layout.is_u4 = ܫܪܪܐ;
        اكسر;

    حالة GGML_TYPE_Q8_0:
        layout.is_symmetric = ܫܪܪܐ;
        اكسر;

    حالة GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        اكسر;

    حالة GGML_TYPE_Q6_K:
        layout.weights_per_block = 16;
        layout.is_symmetric = ܫܪܪܐ;
        اكسر;

    حالة GGML_TYPE_Q5_K:
        اكسر;

    افتراضي:
        // Unsupported quantization type
        ܦܢܐ layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_blocks = n_elements / layout.weights_per_block;
    layout.scales_size = n_blocks * (tensor->type == GGML_TYPE_MXFP4 ? حجم(uint8_t) : حجم(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    ܐܢ (layout.is_symmetric) {
        layout.zp_size = 0;
    } ܐܚܪܢܐ ܐܢ (use_bias || for_gather_matmul) {
        layout.zp_size = n_blocks * حجم(uint16_t);
    } ܐܚܪܢܐ {
        layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_offset = 0;
    layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
    layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
    layout.total_size = layout.zp_offset + layout.zp_size;
    layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));

    ܦܢܐ layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(ثابت ggml_tensor * tensor, منطقي is_remote) {
    ov::Shape shape;
    لكل (صحيح i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    بدل (tensor->type) {
    حالة GGML_TYPE_F32:
        element_type = ov::element::f32;
        اكسر;
    حالة GGML_TYPE_F16:
        element_type = ov::element::f16;
        اكسر;
    حالة GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        اكسر;
    حالة GGML_TYPE_I32:
        element_type = ov::element::i32;
        اكسر;
    حالة GGML_TYPE_I64:
        element_type = ov::element::i64;
        اكسر;
    افتراضي:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        ܦܢܐ nullptr;
    }

    auto remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    ܐܢ (is_remote) {
        GGML_ASSERT(ggml_openvino_is_gpu());
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        auto usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->data);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } ܐܚܪܢܐ {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->data);
    }

    ܦܢܐ جديد ggml_openvino_tensor_extra(ov_tensor);
}
