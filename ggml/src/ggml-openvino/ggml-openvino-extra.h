#pragma once

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "openvino/runtime/core.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#अन्तर्भाव <CL/cl.h>

#अन्तर्भाव <cstdlib>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/runtime/remote_context.hpp>
#अन्तर्भाव <openvino/runtime/tensor.hpp>
#अन्तर्भाव <optional>
#अन्तर्भाव <string>

// ExtraQuantType enum - defines requantization target formats
// Q4_1_64: u4, group 64, *true* asymmetric (per-group scale and zero point). Note that
// Q4_0_128/Q4_0_C are symmetric despite taking the unsigned branch of quantize_q4_0 -- that branch
// pins zp to 8 with d = max/-8, which is algebraically symmetric.
गणना class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q4_0_64, Q8_0_C, Q8_0_32, Q4_1_64 };

ov::Core & ov_singleton_core();

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context();

// Get the compile config for the current device
ध्रुव ov::AnyMap & ggml_openvino_get_compile_config();

// Get the OpenCL command queue for GPU operations (returns nullptr for CPU/NPU)
cl_command_queue ggml_openvino_get_cl_queue();

// Intel USM extension function type
प्रकारनिर्देश cl_int(CL_API_CALL * clEnqueueMemFillINTEL_fn)(cl_command_queue queue,
                                                       शून्य * dst_ptr,
                                                       ध्रुव शून्य * pattern,
                                                       size_t pattern_size,
                                                       size_t size,
                                                       cl_uint अंक_events_in_wait_list,
                                                       ध्रुव cl_event * event_wait_list,
                                                       cl_event * event);

प्रकारनिर्देश cl_int(CL_API_CALL * clEnqueueMemcpyINTEL_fn)(cl_command_queue queue,
                                                      cl_bool blocking,
                                                      शून्य * dst_ptr,
                                                      ध्रुव शून्य * src_ptr,
                                                      size_t size,
                                                      cl_uint अंक_events_in_wait_list,
                                                      ध्रुव cl_event * event_wait_list,
                                                      cl_event * event);

// Get the clEnqueueMemFillINTEL function pointer (returns nullptr if not available)
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL();

// Get the clEnqueueMemcpyINTEL function pointer (returns nullptr if not available)
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL();

// =====================================================
// Global Device Configuration (singleton)
// =====================================================
// Initialized once during backend init from GGML_OPENVINO_DEVICE env var

संरचना ggml_openvino_device_config {
    std::string device_name = "CPU";
    std::vector<std::string> available_devices;
    bool is_npu = false;
    bool initialized = false;
    std::optional<ov::RemoteContext> remote_context;
    size_t max_alloc_size = SIZE_MAX;
    ov::AnyMap compile_config;
    std::unordered_map<std::string, std::string> environment_variables;
    cl_command_queue cl_queue = nullptr;
    clEnqueueMemFillINTEL_fn cl_mem_fill_fn = nullptr;
    clEnqueueMemcpyINTEL_fn cl_mem_cpy_fn = nullptr;

    शून्य init();
    ~ggml_openvino_device_config();
};

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config();

// Initialize device config (call during backend init)
शून्य ggml_openvino_init_device_config();

// Get the device name
ध्रुव std::string & ggml_openvino_get_device_name();

// Get all available physical OpenVINO devices
std::vector<std::string> ggml_openvino_get_available_devices();

// Human-readable device name, e.g. "Intel(R) AI Boost (NPU 4000)"; the device id if unavailable
std::string ggml_openvino_get_device_description(ध्रुव std::string & device_name);

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
ध्रुव वर्ण * ggml_openvino_getenv_str(ध्रुव वर्ण * var, ध्रुव वर्ण * default_मूल्य = nullptr);
पूर्णाङ्क ggml_openvino_getenv_int(ध्रुव वर्ण * var, पूर्णाङ्क default_मूल्य = 0);

// Memory optimization toggles. GGML_OPENVINO_MEMORY_OPTIMIZE is an umbrella
// switch; the fine-grained env vars still override it when explicitly set.
bool ggml_openvino_reduce_compile_mem_enabled();
bool ggml_openvino_release_weights_enabled();

// Check if running on NPU
bool ggml_openvino_is_npu();

// Check if running on a GPU (GPU, GPU.0, GPU.1, ...)
bool ggml_openvino_is_gpu();

// Largest single memory object the device can allocate, SIZE_MAX when there is no known limit
size_t ggml_openvino_max_alloc_size();

// Host weight-buffer release (GGML_OPENVINO_RELEASE_WEIGHTS, GPU only).
// register: record a host weight buffer (idempotent per data pointer).
// release:  madvise(MADV_DONTNEED) all registered buffers, dropping their RSS.
// released: true once release has run (used to fail-fast on post-release recompile).
शून्य ggml_openvino_register_weight_buffer(शून्य * data, size_t size);
शून्य ggml_openvino_release_weight_buffers();
bool ggml_openvino_weight_buffers_released();

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ध्रुव ggml_tensor * tensor, bool no_requant = false);

// =====================================================
// OpenVINO Tensor Extra Types
// =====================================================
// These types are stored in tensor->extra by the OpenVINO backend buffer.
// They allow:
// 1. Pre-built ov::Constant nodes for weights (avoiding memcpy during graph construction)
// 2. ov::Tensor wrappers for KV cache / compute tensors (for direct use with infer_request)

// Base class for OpenVINO tensor extra data
संरचना ggml_openvino_extra_base {
    गणना class Type { WEIGHT, QUANTIZED_WEIGHT, TENSOR };
    Type type;
    आभास ~ggml_openvino_extra_base() = स्वभाव;
रक्षित:
    explicit ggml_openvino_extra_base(Type t) : type(t) {}
};

// Extra data for F16/F32/BF16 weight tensors - stores the pre-built weight node
संरचना ggml_openvino_weight_extra : सार्वजनिक ggml_openvino_extra_base {
    ov::Tensor weights;                     // The underlying weight data tensor
    std::shared_ptr<ov::गाँठ_2> weight_गाँठ;  // Pre-built OpenVINO weight node

    ggml_openvino_weight_extra(ov::Tensor w, std::shared_ptr<ov::गाँठ_2> n) :
        ggml_openvino_extra_base(Type::WEIGHT),
        weights(std::move(w)),
        weight_गाँठ(std::move(n)) {}
};

// Extra data for quantized weight tensors - stores extracted weights/scales/zp and weight node
संरचना ggml_openvino_quantized_weight_extra : सार्वजनिक ggml_openvino_extra_base {
    ov::Tensor weights;                     // U4 or U8 extracted weights
    ov::Tensor scales;                      // F16 scales
    ov::Tensor zp;                          // U4 or U8 zero points (same type as weights)
    std::shared_ptr<ov::गाँठ_2> weight_गाँठ;  // Pre-built OpenVINO weight subgraph

    ggml_openvino_quantized_weight_extra(ov::Tensor w, ov::Tensor s, ov::Tensor z, std::shared_ptr<ov::गाँठ_2> n) :
        ggml_openvino_extra_base(Type::QUANTIZED_WEIGHT),
        weights(std::move(w)),
        scales(std::move(s)),
        zp(std::move(z)),
        weight_गाँठ(std::move(n)) {}
};

// Extra data for KV cache / compute tensors - stores ov::Tensor for infer_request
संरचना ggml_openvino_tensor_extra : सार्वजनिक ggml_openvino_extra_base {
    std::shared_ptr<ov::Tensor> tensor;  // For direct use with infer_request

    explicit ggml_openvino_tensor_extra(std::shared_ptr<ov::Tensor> t) :
        ggml_openvino_extra_base(Type::TENSOR),
        tensor(std::move(t)) {}
};

// =====================================================
// Extracted Size Calculation for Quantized Tensors
// =====================================================
// For quantized tensors, we need extra space to store extracted weights, scales, and zero points.
// Returns the total size needed in the buffer for extracted data.

संरचना ggml_openvino_extracted_layout {
    size_t total_size = 0;      // Total bytes needed
    size_t weights_offset = 0;  // Offset to weights in buffer
    size_t weights_size = 0;    // Size of weights in bytes
    size_t scales_offset = 0;   // Offset to scales in buffer
    size_t scales_size = 0;     // Size of scales in bytes
    size_t zp_offset = 0;       // Offset to zero points in buffer
    size_t zp_size = 0;         // Size of zero points in bytes (U4 or U8)
    bool is_u4;                 // true for U4 weights, false for U8
    int64_t weights_per_block;  // weights per scale/zp block
    bool is_symmetric;          // true for symmetric quantization

    // Requantization info
    bool is_requant = false;                     // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;  // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(ध्रुव ggml_tensor * tensor, bool use_bias = false);

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(ध्रुव ggml_tensor * tensor, bool is_remote);

// Check if a tensor's buffer uses remote (device) memory (e.g. GPU USM)
bool ggml_openvino_buffer_is_remote(ध्रुव ggml_tensor * tensor);

// Register an extra with the tensor's OpenVINO buffer context for proper lifetime management.
// This sets tensor->extra and tracks the extra in the buffer context for cleanup.
शून्य ggml_openvino_buffer_register_extra(ggml_tensor * tensor, ggml_openvino_extra_base * extra);

// =====================================================
// OpenVINO Backend Context and Interface
// =====================================================
संरचना ggml_backend_openvino_context {
    पूर्णाङ्क device = 0;
    std::string name = "OpenVINO";
    std::string description = "OpenVINO Backend Context";

    std::shared_ptr<शून्य> runtime_context = nullptr;

    ggml_backend_openvino_context() = स्वभाव;
};
