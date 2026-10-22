#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "openvino/runtime/core.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು CL_ಗುರಿ_OPENCL_ಆವೃತ್ತಿ 300
#ಅನ್ತರ್ಭಾವ <CL/cl.h>

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/remote_context.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/tensor.hpp>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <string>

// ExtraQuantType enum - defines requantization target formats
// Q4_1_64: u4, group 64, *true* asymmetric (per-group scale and zero point). Note that
// Q4_0_128/Q4_0_C are symmetric despite taking the unsigned branch of quantize_q4_0 -- that branch
// pins zp to 8 with d = max/-8, which is algebraically symmetric.
ಎಣಿಕೆ class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q4_0_64, Q8_0_C, Q8_0_32, Q4_1_64 };

ov::Core & ov_singleton_core();

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context();

// Get the compile config for the current device
ಸ್ಥಿರಾಂಕ ov::AnyMap & ggml_openvino_get_compile_config();

// Get the OpenCL command queue for GPU operations (returns nullptr for CPU/NPU)
cl_command_ಸರತಿ ggml_openvino_get_cl_ಸರತಿ();

// Intel USM extension function type
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ cl_int(CL_API_CALL * cl_ಸರತಿಗೆ_ಸೇರಿಸು_Mem_Fill_INTEL_fn)(cl_command_ಸರತಿ ಸರತಿ,
                                                       ಶೂನ್ಯ * dst_ptr,
                                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ನಮೂನೆ,
                                                       size_t ನಮೂನೆ_ಗಾತ್ರ,
                                                       size_t ಗಾತ್ರ_2,
                                                       cl_uint ಅಂಕೆ_events_ಒಳ_wait_list,
                                                       ಸ್ಥಿರಾಂಕ cl_event * event_wait_list,
                                                       cl_event * event);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ cl_int(CL_API_CALL * cl_ಸರತಿಗೆ_ಸೇರಿಸು_Memcpy_INTEL_fn)(cl_command_ಸರತಿ ಸರತಿ,
                                                      cl_bool blocking,
                                                      ಶೂನ್ಯ * dst_ptr,
                                                      ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src_ptr,
                                                      size_t ಗಾತ್ರ_2,
                                                      cl_uint ಅಂಕೆ_events_ಒಳ_wait_list,
                                                      ಸ್ಥಿರಾಂಕ cl_event * event_wait_list,
                                                      cl_event * event);

// Get the clEnqueueMemFillINTEL function pointer (returns nullptr if not available)
cl_ಸರತಿಗೆ_ಸೇರಿಸು_Mem_Fill_INTEL_fn ggml_openvino_get_cl_ಸರತಿಗೆ_ಸೇರಿಸು_Mem_Fill_INTEL();

// Get the clEnqueueMemcpyINTEL function pointer (returns nullptr if not available)
cl_ಸರತಿಗೆ_ಸೇರಿಸು_Memcpy_INTEL_fn ggml_openvino_get_cl_ಸರತಿಗೆ_ಸೇರಿಸು_Memcpy_INTEL();

// =====================================================
// Global Device Configuration (singleton)
// =====================================================
// Initialized once during backend init from GGML_OPENVINO_DEVICE env var

ರಚನೆ ggml_openvino_device_config {
    std::string device_name = "CPU";
    std::ಸದಿಶ<std::string> ಲಭ್ಯ_devices;
    bool is_npu = false;
    bool initialized = false;
    std::optional<ov::RemoteContext> remote_context;
    size_t max_alloc_ಗಾತ್ರ = SIZE_MAX;
    ov::AnyMap compile_config;
    std::unordered_map<std::string, std::string> environment_variables;
    cl_command_ಸರತಿ cl_ಸರತಿ = nullptr;
    cl_ಸರತಿಗೆ_ಸೇರಿಸು_Mem_Fill_INTEL_fn cl_mem_fill_fn = nullptr;
    cl_ಸರತಿಗೆ_ಸೇರಿಸು_Memcpy_INTEL_fn cl_mem_cpy_fn = nullptr;

    ಶೂನ್ಯ init();
    ~ggml_openvino_device_config();
};

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config();

// Initialize device config (call during backend init)
ಶೂನ್ಯ ggml_openvino_init_device_config();

// Get the device name
ಸ್ಥಿರಾಂಕ std::string & ggml_openvino_get_device_name();

// Get all available physical OpenVINO devices
std::ಸದಿಶ<std::string> ggml_openvino_get_ಲಭ್ಯ_devices();

// Human-readable device name, e.g. "Intel(R) AI Boost (NPU 4000)"; the device id if unavailable
std::string ggml_openvino_get_device_description(ಸ್ಥಿರಾಂಕ std::string & device_name);

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
ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_openvino_getenv_str(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * var, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * default_ಮೌಲ್ಯ = nullptr);
ಪೂರ್ಣಾಂಕ ggml_openvino_getenv_int(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * var, ಪೂರ್ಣಾಂಕ default_ಮೌಲ್ಯ = 0);

// Memory optimization toggles. GGML_OPENVINO_MEMORY_OPTIMIZE is an umbrella
// switch; the fine-grained env vars still override it when explicitly set.
bool ggml_openvino_reduce_compile_mem_enabled();
bool ggml_openvino_release_weights_enabled();

// Check if running on NPU
bool ggml_openvino_is_npu();

// Check if running on a GPU (GPU, GPU.0, GPU.1, ...)
bool ggml_openvino_is_gpu();

// Largest single memory object the device can allocate, SIZE_MAX when there is no known limit
size_t ggml_openvino_max_alloc_ಗಾತ್ರ();

// Host weight-buffer release (GGML_OPENVINO_RELEASE_WEIGHTS, GPU only).
// register: record a host weight buffer (idempotent per data pointer).
// release:  madvise(MADV_DONTNEED) all registered buffers, dropping their RSS.
// released: true once release has run (used to fail-fast on post-release recompile).
ಶೂನ್ಯ ggml_openvino_ರಿಜಿಸ್ಟರ್_ತೂಕ_ಬಫರ್(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_openvino_release_ತೂಕ_buffers();
bool ggml_openvino_ತೂಕ_buffers_released();

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, bool no_requant = false);

// =====================================================
// OpenVINO Tensor Extra Types
// =====================================================
// These types are stored in tensor->extra by the OpenVINO backend buffer.
// They allow:
// 1. Pre-built ov::Constant nodes for weights (avoiding memcpy during graph construction)
// 2. ov::Tensor wrappers for KV cache / compute tensors (for direct use with infer_request)

// Base class for OpenVINO tensor extra data
ರಚನೆ ggml_openvino_extra_ತಳ {
    ಎಣಿಕೆ class Type { ತೂಕ_2, QUANTIZED_ತೂಕ, TENSOR };
    Type type;
    ವಾಸ್ತವಿಕ ~ggml_openvino_extra_ತಳ() = ಪೂರ್ವನಿಯೋಜಿತ;
ಸಂರಕ್ಷಿತ:
    explicit ggml_openvino_extra_ತಳ(Type t) : type(t) {}
};

// Extra data for F16/F32/BF16 weight tensors - stores the pre-built weight node
ರಚನೆ ggml_openvino_ತೂಕ_extra : ಸಾರ್ವಜನಿಕ ggml_openvino_extra_ತಳ {
    ov::Tensor weights;                     // The underlying weight data tensor
    std::shared_ptr<ov::ಗಂಟು_2> ತೂಕ_ಗಂಟು;  // Pre-built OpenVINO weight node

    ggml_openvino_ತೂಕ_extra(ov::Tensor w, std::shared_ptr<ov::ಗಂಟು_2> n) :
        ggml_openvino_extra_ತಳ(Type::ತೂಕ_2),
        weights(std::move(w)),
        ತೂಕ_ಗಂಟು(std::move(n)) {}
};

// Extra data for quantized weight tensors - stores extracted weights/scales/zp and weight node
ರಚನೆ ggml_openvino_quantized_ತೂಕ_extra : ಸಾರ್ವಜನಿಕ ggml_openvino_extra_ತಳ {
    ov::Tensor weights;                     // U4 or U8 extracted weights
    ov::Tensor scales;                      // F16 scales
    ov::Tensor zp;                          // U4 or U8 zero points (same type as weights)
    std::shared_ptr<ov::ಗಂಟು_2> ತೂಕ_ಗಂಟು;  // Pre-built OpenVINO weight subgraph

    ggml_openvino_quantized_ತೂಕ_extra(ov::Tensor w, ov::Tensor s, ov::Tensor z, std::shared_ptr<ov::ಗಂಟು_2> n) :
        ggml_openvino_extra_ತಳ(Type::QUANTIZED_ತೂಕ),
        weights(std::move(w)),
        scales(std::move(s)),
        zp(std::move(z)),
        ತೂಕ_ಗಂಟು(std::move(n)) {}
};

// Extra data for KV cache / compute tensors - stores ov::Tensor for infer_request
ರಚನೆ ggml_openvino_tensor_extra : ಸಾರ್ವಜನಿಕ ggml_openvino_extra_ತಳ {
    std::shared_ptr<ov::Tensor> tensor;  // For direct use with infer_request

    explicit ggml_openvino_tensor_extra(std::shared_ptr<ov::Tensor> t) :
        ggml_openvino_extra_ತಳ(Type::TENSOR),
        tensor(std::move(t)) {}
};

// =====================================================
// Extracted Size Calculation for Quantized Tensors
// =====================================================
// For quantized tensors, we need extra space to store extracted weights, scales, and zero points.
// Returns the total size needed in the buffer for extracted data.

ರಚನೆ ggml_openvino_extracted_layout {
    size_t ಒಟ್ಟು_ಗಾತ್ರ = 0;      // Total bytes needed
    size_t weights_ಸ್ಥಾನಾಂತರ = 0;  // Offset to weights in buffer
    size_t weights_ಗಾತ್ರ = 0;    // Size of weights in bytes
    size_t scales_ಸ್ಥಾನಾಂತರ = 0;   // Offset to scales in buffer
    size_t scales_ಗಾತ್ರ = 0;     // Size of scales in bytes
    size_t zp_ಸ್ಥಾನಾಂತರ = 0;       // Offset to zero points in buffer
    size_t zp_ಗಾತ್ರ = 0;         // Size of zero points in bytes (U4 or U8)
    bool is_u4;                 // true for U4 weights, false for U8
    int64_t weights_ಪ್ರತಿ_ಖಂಡ;  // weights per scale/zp block
    bool is_ಸಮಮಿತ;          // true for symmetric quantization

    // Requantization info
    bool is_requant = false;                     // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;  // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, bool use_bias = false);

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, bool is_remote);

// Check if a tensor's buffer uses remote (device) memory (e.g. GPU USM)
bool ggml_openvino_ಬಫರ್_is_remote(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);

// Register an extra with the tensor's OpenVINO buffer context for proper lifetime management.
// This sets tensor->extra and tracks the extra in the buffer context for cleanup.
ಶೂನ್ಯ ggml_openvino_ಬಫರ್_ರಿಜಿಸ್ಟರ್_extra(ggml_tensor * tensor, ggml_openvino_extra_ತಳ * extra);

// =====================================================
// OpenVINO Backend Context and Interface
// =====================================================
ರಚನೆ ggml_backend_openvino_context {
    ಪೂರ್ಣಾಂಕ device = 0;
    std::string name = "OpenVINO";
    std::string description = "OpenVINO Backend Context";

    std::shared_ptr<ಶೂನ್ಯ> runtime_context = nullptr;

    ggml_backend_openvino_context() = ಪೂರ್ವನಿಯೋಜಿತ;
};
