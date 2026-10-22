#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/runtime/core.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CL_ਨਿਸ਼ਾਨਾ_OPENCL_ਸੰਸਕਰਣ 300
#ਅਨ੍ਤਰ੍ਭਾਵ <CL/cl.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/remote_context.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/tensor.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

// ExtraQuantType enum - defines requantization target formats
// Q4_1_64: u4, group 64, *true* asymmetric (per-group scale and zero point). Note that
// Q4_0_128/Q4_0_C are symmetric despite taking the unsigned branch of quantize_q4_0 -- that branch
// pins zp to 8 with d = max/-8, which is algebraically symmetric.
ਗਿਣਤੀ class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q4_0_64, Q8_0_C, Q8_0_32, Q4_1_64 };

ov::Core & ov_singleton_core();

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context();

// Get the compile config for the current device
ਅਚਲ ov::AnyMap & ggml_openvino_get_compile_config();

// Get the OpenCL command queue for GPU operations (returns nullptr for CPU/NPU)
cl_command_ਕਤਾਰ ggml_openvino_get_cl_ਕਤਾਰ();

// Intel USM extension function type
ਕਿਸਮਪਰਿਭਾਸ਼ਾ cl_int(CL_API_CALL * cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Mem_Fill_INTEL_fn)(cl_command_ਕਤਾਰ ਕਤਾਰ,
                                                       ਖਾਲੀ * dst_ptr,
                                                       ਅਚਲ ਖਾਲੀ * ਨਮੂਨਾ,
                                                       size_t ਨਮੂਨਾ_ਮਾਪ,
                                                       size_t ਮਾਪ,
                                                       cl_uint ਅੰਕ_events_ਵਿੱਚ_wait_list,
                                                       ਅਚਲ cl_event * event_wait_list,
                                                       cl_event * event);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ cl_int(CL_API_CALL * cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Memcpy_INTEL_fn)(cl_command_ਕਤਾਰ ਕਤਾਰ,
                                                      cl_bool blocking,
                                                      ਖਾਲੀ * dst_ptr,
                                                      ਅਚਲ ਖਾਲੀ * src_ptr,
                                                      size_t ਮਾਪ,
                                                      cl_uint ਅੰਕ_events_ਵਿੱਚ_wait_list,
                                                      ਅਚਲ cl_event * event_wait_list,
                                                      cl_event * event);

// Get the clEnqueueMemFillINTEL function pointer (returns nullptr if not available)
cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Mem_Fill_INTEL_fn ggml_openvino_get_cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Mem_Fill_INTEL();

// Get the clEnqueueMemcpyINTEL function pointer (returns nullptr if not available)
cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Memcpy_INTEL_fn ggml_openvino_get_cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Memcpy_INTEL();

// =====================================================
// Global Device Configuration (singleton)
// =====================================================
// Initialized once during backend init from GGML_OPENVINO_DEVICE env var

ਬਣਤਰ ggml_openvino_device_config {
    std::string device_name = "CPU";
    std::ਸਦਿਸ਼<std::string> ਉਪਲਬਧ_devices;
    bool is_npu = false;
    bool initialized = false;
    std::optional<ov::RemoteContext> remote_context;
    size_t max_alloc_ਮਾਪ = SIZE_MAX;
    ov::AnyMap compile_config;
    std::unordered_map<std::string, std::string> environment_variables;
    cl_command_ਕਤਾਰ cl_ਕਤਾਰ = nullptr;
    cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Mem_Fill_INTEL_fn cl_mem_fill_fn = nullptr;
    cl_ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_Memcpy_INTEL_fn cl_mem_cpy_fn = nullptr;

    ਖਾਲੀ init();
    ~ggml_openvino_device_config();
};

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config();

// Initialize device config (call during backend init)
ਖਾਲੀ ggml_openvino_init_device_config();

// Get the device name
ਅਚਲ std::string & ggml_openvino_get_device_name();

// Get all available physical OpenVINO devices
std::ਸਦਿਸ਼<std::string> ggml_openvino_get_ਉਪਲਬਧ_devices();

// Human-readable device name, e.g. "Intel(R) AI Boost (NPU 4000)"; the device id if unavailable
std::string ggml_openvino_get_device_description(ਅਚਲ std::string & device_name);

// Environment variable accessors. All GGML_OPENVINO_* env vars are read once
// during backend init and cached on the device config; consumers must go
// through these helpers (never call ::getenv directly) so behavior stays
// consistent and centralized.
//
// Use ggml_openvino_getenv_str() for string / path values
// (e.g. GGML_OPENVINO_DEVICE, GGML_OPENVINO_CACHE_DIR). The optional
// default_value is returned when the var is unset or empty.
//
// Use ggml_openvino_getenv_int() for boolean toggles and integer settings.
// It returns std::atoi(value) when set, otherwise default_value. For
// boolean use, `if (ggml_openvino_getenv_int(name))` is true iff the value
// is a non-zero integer (so "0" disables, "1" enables).
ਅਚਲ ਅੱਖਰ * ggml_openvino_getenv_str(ਅਚਲ ਅੱਖਰ * var, ਅਚਲ ਅੱਖਰ * default_ਮੁੱਲ = nullptr);
ਪੂਰਨ ggml_openvino_getenv_int(ਅਚਲ ਅੱਖਰ * var, ਪੂਰਨ default_ਮੁੱਲ = 0);

// Memory optimization toggles. GGML_OPENVINO_MEMORY_OPTIMIZE is an umbrella
// switch; the fine-grained env vars still override it when explicitly set.
bool ggml_openvino_reduce_compile_mem_enabled();
bool ggml_openvino_release_weights_enabled();

// Check if running on NPU
bool ggml_openvino_is_npu();

// Check if running on a GPU (GPU, GPU.0, GPU.1, ...)
bool ggml_openvino_is_gpu();

// Largest single memory object the device can allocate, SIZE_MAX when there is no known limit
size_t ggml_openvino_max_alloc_ਮਾਪ();

// Host weight-buffer release (GGML_OPENVINO_RELEASE_WEIGHTS, GPU only).
// register: record a host weight buffer (idempotent per data pointer).
// release:  madvise(MADV_DONTNEED) all registered buffers, dropping their RSS.
// released: true once release has run (used to fail-fast on post-release recompile).
ਖਾਲੀ ggml_openvino_ਰਜਿਸਟਰ_ਭਾਰ_ਬਫਰ(ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ);
ਖਾਲੀ ggml_openvino_release_ਭਾਰ_buffers();
bool ggml_openvino_ਭਾਰ_buffers_released();

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ਅਚਲ ggml_tensor * tensor, bool no_requant = false);

// =====================================================
// OpenVINO Tensor Extra Types
// =====================================================
// These types are stored in tensor->extra by the OpenVINO backend buffer.
// They allow:
// 1. Pre-built ov::Constant nodes for weights (avoiding memcpy during graph construction)
// 2. ov::Tensor wrappers for KV cache / compute tensors (for direct use with infer_request)

// Base class for OpenVINO tensor extra data
ਬਣਤਰ ggml_openvino_extra_ਅਧਾਰ {
    ਗਿਣਤੀ class Type { ਭਾਰ_2, QUANTIZED_ਭਾਰ, TENSOR };
    Type type;
    ਆਭਾਸੀ ~ggml_openvino_extra_ਅਧਾਰ() = ਮੂਲ;
ਸੁਰੱਖਿਅਤ:
    explicit ggml_openvino_extra_ਅਧਾਰ(Type t) : type(t) {}
};

// Extra data for F16/F32/BF16 weight tensors - stores the pre-built weight node
ਬਣਤਰ ggml_openvino_ਭਾਰ_extra : ਜਨਤਕ ggml_openvino_extra_ਅਧਾਰ {
    ov::Tensor weights;                     // The underlying weight data tensor
    std::shared_ptr<ov::ਗੰਢ_2> ਭਾਰ_ਗੰਢ;  // Pre-built OpenVINO weight node

    ggml_openvino_ਭਾਰ_extra(ov::Tensor w, std::shared_ptr<ov::ਗੰਢ_2> n) :
        ggml_openvino_extra_ਅਧਾਰ(Type::ਭਾਰ_2),
        weights(std::move(w)),
        ਭਾਰ_ਗੰਢ(std::move(n)) {}
};

// Extra data for quantized weight tensors - stores extracted weights/scales/zp and weight node
ਬਣਤਰ ggml_openvino_quantized_ਭਾਰ_extra : ਜਨਤਕ ggml_openvino_extra_ਅਧਾਰ {
    ov::Tensor weights;                     // U4 or U8 extracted weights
    ov::Tensor scales;                      // F16 scales
    ov::Tensor zp;                          // U4 or U8 zero points (same type as weights)
    std::shared_ptr<ov::ਗੰਢ_2> ਭਾਰ_ਗੰਢ;  // Pre-built OpenVINO weight subgraph

    ggml_openvino_quantized_ਭਾਰ_extra(ov::Tensor w, ov::Tensor s, ov::Tensor z, std::shared_ptr<ov::ਗੰਢ_2> n) :
        ggml_openvino_extra_ਅਧਾਰ(Type::QUANTIZED_ਭਾਰ),
        weights(std::move(w)),
        scales(std::move(s)),
        zp(std::move(z)),
        ਭਾਰ_ਗੰਢ(std::move(n)) {}
};

// Extra data for KV cache / compute tensors - stores ov::Tensor for infer_request
ਬਣਤਰ ggml_openvino_tensor_extra : ਜਨਤਕ ggml_openvino_extra_ਅਧਾਰ {
    std::shared_ptr<ov::Tensor> tensor;  // For direct use with infer_request

    explicit ggml_openvino_tensor_extra(std::shared_ptr<ov::Tensor> t) :
        ggml_openvino_extra_ਅਧਾਰ(Type::TENSOR),
        tensor(std::move(t)) {}
};

// =====================================================
// Extracted Size Calculation for Quantized Tensors
// =====================================================
// For quantized tensors, we need extra space to store extracted weights, scales, and zero points.
// Returns the total size needed in the buffer for extracted data.

ਬਣਤਰ ggml_openvino_extracted_layout {
    size_t ਕੁੱਲ_ਮਾਪ = 0;      // Total bytes needed
    size_t weights_ਵਿੱਥ = 0;  // Offset to weights in buffer
    size_t weights_ਮਾਪ = 0;    // Size of weights in bytes
    size_t scales_ਵਿੱਥ = 0;   // Offset to scales in buffer
    size_t scales_ਮਾਪ = 0;     // Size of scales in bytes
    size_t zp_ਵਿੱਥ = 0;       // Offset to zero points in buffer
    size_t zp_ਮਾਪ = 0;         // Size of zero points in bytes (U4 or U8)
    bool is_u4;                 // true for U4 weights, false for U8
    int64_t weights_ਪ੍ਰਤੀ_ਖੰਡ;  // weights per scale/zp block
    bool is_ਸਮਮਿਤ;          // true for symmetric quantization

    // Requantization info
    bool is_requant = false;                     // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;  // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(ਅਚਲ ggml_tensor * tensor, bool use_bias = false);

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(ਅਚਲ ggml_tensor * tensor, bool is_remote);

// Check if a tensor's buffer uses remote (device) memory (e.g. GPU USM)
bool ggml_openvino_ਬਫਰ_is_remote(ਅਚਲ ggml_tensor * tensor);

// Register an extra with the tensor's OpenVINO buffer context for proper lifetime management.
// This sets tensor->extra and tracks the extra in the buffer context for cleanup.
ਖਾਲੀ ggml_openvino_ਬਫਰ_ਰਜਿਸਟਰ_extra(ggml_tensor * tensor, ggml_openvino_extra_ਅਧਾਰ * extra);

// =====================================================
// OpenVINO Backend Context and Interface
// =====================================================
ਬਣਤਰ ggml_backend_openvino_context {
    ਪੂਰਨ device = 0;
    std::string name = "OpenVINO";
    std::string description = "OpenVINO Backend Context";

    std::shared_ptr<ਖਾਲੀ> runtime_context = nullptr;

    ggml_backend_openvino_context() = ਮੂਲ;
};
