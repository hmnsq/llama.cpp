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
    أرجع core;
}

ساكن منطقي has_اڳياڙي(ثابت std::string & s, ثابت std::string & prefix) {
    أرجع s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

ساكن منطقي is_virtual_routing_device(ثابت std::string & device_name) {
    أرجع has_اڳياڙي(device_name, "AUTO") || has_اڳياڙي(device_name, "MULTI") || has_اڳياڙي(device_name, "HETERO");
}

ساكن std::vector<std::string> ov_enumerate_devices() {
    std::vector<std::string> نتيجو;

    لكل (ثابت خودڪار & device : ov_singleton_core().get_موجود_devices()) {
        إذا (!is_virtual_routing_device(device)) {
            نتيجو.push_back(device);
        }
    }

    إذا (نتيجو.خالي_جايون()) {
        نتيجو.push_back("CPU");
    }

    std::sort(نتيجو.begin(), نتيجو.end());
    نتيجو.erase(std::unique(نتيجو.begin(), نتيجو.end()), نتيجو.end());
    أرجع نتيجو;
}

std::string ggml_openvino_get_device_description(ثابت std::string & device_name) {
    std::string description = device_name;
    حاول {
        description = ov_singleton_core().get_property(device_name, ov::device::ڀريل_name);
    } امسك (...) {
        أرجع device_name;
    }

    إذا (has_اڳياڙي(device_name, "NPU")) {
        حاول {
            ثابت std::string arch = ov_singleton_core().get_property(device_name, "DEVICE_ARCHITECTURE").as<std::string>();
            إذا (!arch.خالي_جايون()) {
                description += " (NPU " + arch + ")";
            }
        } امسك (...) {
        }
    }

    أرجع description;
}

// requested: GGML_OPENVINO_DEVICE, nullptr if unset. available_devices is never empty (see ov_enumerate_devices)
ساكن std::string resolve_openvino_device_name(ثابت std::vector<std::string> & موجود_devices,
                                                ثابت حرف * requested) {
    خودڪار موجود = [&](ثابت std::string & name) {
        أرجع std::find(موجود_devices.begin(), موجود_devices.end(), name) != موجود_devices.end();
    };
    إذا (requested == خالي_اشارو) {
        أرجع موجود("CPU") ? "CPU" : موجود_devices.front();
    }
    إذا (!موجود(requested)) {
        // No fallback to CPU (easy to miss) and no GPU -> GPU.0 alias (with iGPU + dGPU, GPU.0 is often the
        // wrong one). List the devices here: --list-devices initializes this backend and would abort too.
        std::string list;
        لكل (ثابت std::string & name : موجود_devices) {
            list += "\n  " + name + ": " + ggml_openvino_get_device_description(name);
        }
        GGML_ABORT("GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s is not available. "
                   "Set it to one of the available OpenVINO devices:%s",
                   requested, list.c_str());
    }
    أرجع requested;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

فراغ ggml_openvino_device_config::init() {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> تالو(mutex);
    إذا (initialized) {
        أرجع;
    }
    // Set up front: a failed OpenCL setup below is not retried on every call
    initialized = سچ;

    // All recognized GGML_OPENVINO_* env vars. Their values are cached here
    // once at backend init time and read back via ggml_openvino_getenv_str()
    // (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
    ساكن constexpr ثابت حرف * env_var_نالا[] = {
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

    لكل (ثابت حرف * ثابت & env_var : env_var_نالا) {
        خودڪار * env = getenv(env_var);
        إذا (env) {
            environment_variables[env_var] = env;
        }
    }

    موجود_devices = ov_enumerate_devices();
    device_name = resolve_openvino_device_name(موجود_devices, ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE"));
    is_npu = has_اڳياڙي(device_name, "NPU");

    ggml_openvino_model_cache_init();

    ثابت حرف * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    إذا (has_اڳياڙي(device_name, "NPU")) {
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
        إذا (cache_dir && لڙي_ڊيگهه(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_گهڻائي(ov::Cache_گهڻائي::OPTIMIZE_سائيز));
        }
        ثابت حرف * compilation_گهڻائي_params =
            ggml_openvino_getenv_str("GGML_OPENVINO_NPU_COMPILE_CONFIG");
        إذا (compilation_گهڻائي_params && لڙي_ڊيگهه(compilation_گهڻائي_params) > 0) {
            compile_config["NPU_COMPILATION_MODE_PARAMS"] = compilation_گهڻائي_params;
        }
    } وإلا إذا (cache_dir && لڙي_ڊيگهه(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_گهڻائي(ov::Cache_گهڻائي::OPTIMIZE_سائيز));
    }

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2) {
        compile_config.insert(ov::enable_profiling(سچ));
    }

    // Initialize remote context with queue sharing for GPU
    إذا (has_اڳياڙي(device_name, "GPU")) {
        // Use the OpenCL context OpenVINO created for this device, so GPU.N gets its own device
        cl_context cl_ctx;
        حاول {
            خودڪار ov_ctx = ov_singleton_core().get_default_context(device_name).as<ov::intel_gpu::ocl::ClContext>();
            cl_ctx = ov_ctx.get();
        } امسك (ثابت std::exception & e) {
            // The consumers of the remote context have no host fallback, and OpenVINO
            // already reported the device as present.
            GGML_ABORT("ggml-openvino: failed to get the OpenCL context for %s: %s", device_name.c_str(), e.what());
        }

        cl_int err;
        cl_device_id cl_device;
        err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, حجم(cl_device), &cl_device, خالي_اشارو);
        إذا (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL device for %s: %d", device_name.c_str(), err);
        }

        cl_platform_id cl_platform;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_PLATFORM, حجم(cl_platform), &cl_platform, خالي_اشارو);
        إذا (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to get the OpenCL platform for %s: %d", device_name.c_str(), err);
        }

        cl_mem_fill_fn =
            (cl_قطار_اندر_وجهو_Mem_Fill_INTEL_fn) cl_Get_Extension_Function_پتو_For_Platform(cl_platform, "clEnqueueMemFillINTEL");
        cl_mem_cpy_fn =
            (cl_قطار_اندر_وجهو_Memcpy_INTEL_fn) cl_Get_Extension_Function_پتو_For_Platform(cl_platform, "clEnqueueMemcpyINTEL");

        cl_ulong device_max_alloc = 0;
        err = clGetDeviceInfo(cl_device, CL_DEVICE_MAX_MEM_ALLOC_سائيز, حجم(device_max_alloc), &device_max_alloc,
                              خالي_اشارو);
        إذا (err == CL_SUCCESS) {
            max_alloc_سائيز = device_max_alloc;
        } وإلا {
            // not fatal, ggml then allocates one buffer
            GGML_LOG_WARN("Failed to get OpenCL max allocation size: %d\n", err);
        }

        ثابت cl_قطار_properties profiling_properties[] = {
            CL_قطار_PROPERTIES,
            CL_قطار_PROFILING_ENABLE,
            0,
        };
        ثابت cl_قطار_properties * قطار_properties =
            ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? profiling_properties : خالي_اشارو;
        cl_قطار = cl_Create_Command_قطار_With_Properties(cl_ctx, cl_device, قطار_properties, &err);
        إذا (err != CL_SUCCESS) {
            GGML_ABORT("ggml-openvino: failed to create the OpenCL queue for %s: %d", device_name.c_str(), err);
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_قطار);
    } وإلا إذا (has_اڳياڙي(device_name, "NPU")) {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    إذا (cl_قطار != خالي_اشارو) {
        cl_Release_Command_قطار(cl_قطار);
        cl_قطار = خالي_اشارو;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    ساكن ggml_openvino_device_config config;
    أرجع config;
}

// Initialize device config (call during backend init)
فراغ ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
ثابت std::string & ggml_openvino_get_device_name() {
    أرجع ggml_openvino_get_device_config().device_name;
}

std::vector<std::string> ggml_openvino_get_موجود_devices() {
    خودڪار & config = ggml_openvino_get_device_config();
    config.init();
    أرجع config.موجود_devices;
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
ثابت حرف * ggml_openvino_getenv_str(ثابت حرف * var, ثابت حرف * default_قدر) {
    خودڪار & env_map = ggml_openvino_get_device_config().environment_variables;
    خودڪار it = env_map.find(var);
    أرجع (it == env_map.end() || it->second.خالي_جايون()) ? default_قدر : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int (via std::atoi).
// Returns default_value (0) when the var is unset or empty. Used for both
// integer settings (e.g. GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean
// toggles: "0" disables, any non-zero integer enables.
صحيح ggml_openvino_getenv_int(ثابت حرف * var, صحيح default_قدر) {
    ثابت حرف * v = ggml_openvino_getenv_str(var, خالي_اشارو);
    أرجع v ? std::atoi(v) : default_قدر;
}

منطقي ggml_openvino_reduce_compile_mem_enabled() {
    ثابت حرف * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    إذا (reduce_compile_mem != خالي_اشارو) {
        أرجع ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    أرجع ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

منطقي ggml_openvino_release_weights_enabled() {
    ثابت حرف * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    إذا (release_weights != خالي_اشارو) {
        أرجع ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    أرجع ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
منطقي ggml_openvino_is_npu() {
    أرجع ggml_openvino_get_device_config().is_npu;
}

منطقي ggml_openvino_is_gpu() {
    أرجع has_اڳياڙي(ggml_openvino_get_device_name(), "GPU");
}

size_t ggml_openvino_max_alloc_سائيز() {
    أرجع ggml_openvino_get_device_config().max_alloc_سائيز;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    أرجع ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
ثابت ov::AnyMap & ggml_openvino_get_compile_config() {
    أرجع ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_قطار ggml_openvino_get_cl_قطار() {
    أرجع ggml_openvino_get_device_config().cl_قطار;
}

// Get the clEnqueueMemFillINTEL function pointer
cl_قطار_اندر_وجهو_Mem_Fill_INTEL_fn ggml_openvino_get_cl_قطار_اندر_وجهو_Mem_Fill_INTEL() {
    أرجع ggml_openvino_get_device_config().cl_mem_fill_fn;
}

// Get the clEnqueueMemcpyINTEL function pointer
cl_قطار_اندر_وجهو_Memcpy_INTEL_fn ggml_openvino_get_cl_قطار_اندر_وجهو_Memcpy_INTEL() {
    أرجع ggml_openvino_get_device_config().cl_mem_cpy_fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ثابت ggml_tensor * tensor, منطقي no_requant) {
    إذا (no_requant) {
        أرجع std::nullopt;
    }
    إذا (لڙي_ڀيٽ(tensor->name, "token_embd.weight", 17) == 0) {
        أرجع ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 :
                                                                             ExtraQuantType::Q8_0_C);
    }
    إذا (لڙي_ڀيٽ(tensor->name, "output.weight", 13) == 0) {
        أرجع ExtraQuantType::Q8_0_C;
    }
    إذا (ggml_openvino_is_npu()) {
        أرجع ExtraQuantType::Q4_0_128;
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
    خودڪار is_opt = [rq](ثابت حرف * name) {
        أرجع rq && strcmp(rq, name) == 0;
    };
    ثابت منطقي sym128 = is_opt("q4_sym128");
    ثابت منطقي sym128_all = is_opt("q4_sym128_all");
    ثابت منطقي asym64_all = is_opt("q4_asym64_all");
    ثابت منطقي asym64 = is_opt("q4_asym64");

    إذا (tensor->type == GGML_TYPE_Q4_K) {
        إذا (sym128_all) {
            أرجع ExtraQuantType::Q4_0_128;
        }
        إذا (asym64_all) {
            أرجع ExtraQuantType::Q4_1_64;
        }
    }
    // MoE expert weights (3D, ne[2] = n_expert) stored as Q5_1/Q8_0 are the expert-side
    // equivalent of Q6_K/Q5_K: kept at 8 bits by default while the rest of the model is at 4
    // (gemma-4 26B-A4B keeps its down projection there). Send them to 4 bits under the same
    // option, at group 64 rather than 128: the down expert has k=704, which 64 divides
    // (704/64 = 11) and 128 does not.
    إذا (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        إذا (sym128 || sym128_all) {
            أرجع ExtraQuantType::Q4_0_64;
        }
        إذا (asym64 || asym64_all) {
            أرجع ExtraQuantType::Q4_1_64;
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
        إذا (ggml_openvino_is_gpu() && !is_opt("native")) {
            أرجع ExtraQuantType::Q4_0_64;
        }
    }
    بدل (tensor->type) {
    حالة GGML_TYPE_Q6_K:
    حالة GGML_TYPE_Q5_K:
        إذا (sym128 || sym128_all) {
            أرجع ExtraQuantType::Q4_0_128;
        }
        إذا (asym64 || asym64_all) {
            أرجع ExtraQuantType::Q4_1_64;
        }
        إذا (is_opt("native")) {
            أرجع std::nullopt;
        }
        أرجع ExtraQuantType::Q8_0_C;
    افتراضي:
        أرجع std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(ثابت ggml_tensor * tensor, منطقي use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_متناظر = خطأ;

    إذا (!ggml_is_quantized(tensor->type)) {
        أرجع layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    إذا (tensor->ne[3] != 1) {
        أرجع layout;
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    ثابت منطقي for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    ثابت size_t alignment = 64;  // Good for SIMD

    إذا (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_في_بلاڪ = 32;
        layout.is_متناظر = سچ;
        layout.weights_سائيز = ggml_nbytes(tensor);
        layout.weights_وٿي = 0;
        layout.ڪل_سائيز = layout.weights_سائيز;
        أرجع layout;
    }

    // Check if requantization is needed (NPU-specific)
    خودڪار requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    إذا (requant_type.has_value()) {
        layout.is_requant = سچ;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        إذا (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_سائيز = n_elements * حجم(uint16_t);  // F16 = 2 bytes
            layout.ڪل_سائيز = layout.weights_سائيز;
            layout.weights_وٿي = 0;
            // No scales/zp for F16
            أرجع layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        بدل (requant_type.value()) {
        حالة ExtraQuantType::Q4_0_128:
            layout.is_u4 = سچ;
            layout.weights_في_بلاڪ = 128;
            layout.is_متناظر = سچ;
            اكسر;
        حالة ExtraQuantType::Q4_1_64:
            layout.is_u4 = سچ;
            layout.weights_في_بلاڪ = 64;
            layout.is_متناظر = خطأ;
            اكسر;
        حالة ExtraQuantType::Q4_0_64:
            layout.is_u4 = سچ;
            layout.weights_في_بلاڪ = 64;
            layout.is_متناظر = سچ;
            اكسر;
        حالة ExtraQuantType::Q4_0_C:
            layout.is_u4 = سچ;
            layout.weights_في_بلاڪ = tensor->ne[0];
            layout.is_متناظر = سچ;
            اكسر;
        حالة ExtraQuantType::Q8_0_32:
            layout.is_u4 = خطأ;
            layout.weights_في_بلاڪ = 32;
            layout.is_متناظر = سچ;
            اكسر;
        حالة ExtraQuantType::Q8_0_C:
            layout.is_u4 = خطأ;
            layout.weights_في_بلاڪ = tensor->ne[0];
            layout.is_متناظر = سچ;
            اكسر;
        حالة ExtraQuantType::Q8_1_C:
            layout.is_u4 = خطأ;
            layout.weights_في_بلاڪ = tensor->ne[0];
            اكسر;
        افتراضي:
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }

        إذا (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_سائيز = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_بلاڪون = n_elements / layout.weights_في_بلاڪ;
            layout.scales_سائيز = n_بلاڪون * حجم(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            إذا (layout.is_متناظر) {
                layout.zp_سائيز = 0;
            } وإلا {
                layout.zp_سائيز = layout.is_u4 ? ((n_بلاڪون + 1) / 2) : n_بلاڪون;
            }

            layout.weights_وٿي = 0;
            layout.scales_وٿي = ((layout.weights_سائيز + alignment - 1) / alignment) * alignment;
            layout.zp_وٿي = layout.scales_وٿي + ((layout.scales_سائيز + alignment - 1) / alignment) * alignment;
            layout.ڪل_سائيز = layout.zp_وٿي + layout.zp_سائيز;
            layout.ڪل_سائيز = std::max(layout.ڪل_سائيز, ggml_nbytes(tensor));
            أرجع layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = خطأ;
    layout.weights_في_بلاڪ = 32;
    layout.is_متناظر = خطأ;

    بدل (tensor->type) {
    حالة GGML_TYPE_MXFP4:
    حالة GGML_TYPE_Q4_0:
        layout.is_u4 = سچ;
        layout.is_متناظر = سچ;
        اكسر;

    حالة GGML_TYPE_Q4_1:
    حالة GGML_TYPE_Q4_K:
        layout.is_u4 = سچ;
        اكسر;

    حالة GGML_TYPE_Q8_0:
        layout.is_متناظر = سچ;
        اكسر;

    حالة GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        اكسر;

    حالة GGML_TYPE_Q6_K:
        layout.weights_في_بلاڪ = 16;
        layout.is_متناظر = سچ;
        اكسر;

    حالة GGML_TYPE_Q5_K:
        اكسر;

    افتراضي:
        // Unsupported quantization type
        أرجع layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_سائيز = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_بلاڪون = n_elements / layout.weights_في_بلاڪ;
    layout.scales_سائيز = n_بلاڪون * (tensor->type == GGML_TYPE_MXFP4 ? حجم(uint8_t) : حجم(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    إذا (layout.is_متناظر) {
        layout.zp_سائيز = 0;
    } وإلا إذا (use_bias || for_gather_matmul) {
        layout.zp_سائيز = n_بلاڪون * حجم(uint16_t);
    } وإلا {
        layout.zp_سائيز = layout.is_u4 ? ((n_بلاڪون + 1) / 2) : n_بلاڪون;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_وٿي = 0;
    layout.scales_وٿي = ((layout.weights_سائيز + alignment - 1) / alignment) * alignment;
    layout.zp_وٿي = layout.scales_وٿي + ((layout.scales_سائيز + alignment - 1) / alignment) * alignment;
    layout.ڪل_سائيز = layout.zp_وٿي + layout.zp_سائيز;
    layout.ڪل_سائيز = std::max(layout.ڪل_سائيز, ggml_nbytes(tensor));

    أرجع layout;
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
        أرجع خالي_اشارو;
    }

    خودڪار remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    إذا (is_remote) {
        GGML_ASSERT(ggml_openvino_is_gpu());
        خودڪار gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        خودڪار usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->معلومات);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } وإلا {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->معلومات);
    }

    أرجع جديد ggml_openvino_tensor_extra(ov_tensor);
}
