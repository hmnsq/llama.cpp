#సూచన once

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "openvino/runtime/core.hpp"

#నిర్వచించు CL_లక్ష్యం_OPENCL_సంస్కరణ 300
#అన్తర్భావ <CL/cl.h>

#అన్తర్భావ <cstdlib>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/runtime/remote_context.hpp>
#అన్తర్భావ <openvino/runtime/tensor.hpp>
#అన్తర్భావ <optional>
#అన్తర్భావ <string>

// ExtraQuantType enum - defines requantization target formats
// Q4_1_64: u4, group 64, *true* asymmetric (per-group scale and zero point). Note that
// Q4_0_128/Q4_0_C are symmetric despite taking the unsigned branch of quantize_q4_0 -- that branch
// pins zp to 8 with d = max/-8, which is algebraically symmetric.
గణన class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q4_0_64, Q8_0_C, Q8_0_32, Q4_1_64 };

ov::Core & ov_singleton_core();

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context();

// Get the compile config for the current device
స్థిరరాశి ov::AnyMap & ggml_openvino_get_compile_config();

// Get the OpenCL command queue for GPU operations (returns nullptr for CPU/NPU)
cl_command_వరుస ggml_openvino_get_cl_వరుస();

// Intel USM extension function type
రకనిర్వచనం cl_int(CL_API_CALL * cl_వరుసలో_చేర్చు_Mem_Fill_INTEL_fn)(cl_command_వరుస వరుస,
                                                       శూన్య * dst_ptr,
                                                       స్థిరరాశి శూన్య * నమూనా,
                                                       size_t నమూనా_కొలత,
                                                       size_t కొలత,
                                                       cl_uint అంకె_events_లోపల_wait_list,
                                                       స్థిరరాశి cl_event * event_wait_list,
                                                       cl_event * event);

రకనిర్వచనం cl_int(CL_API_CALL * cl_వరుసలో_చేర్చు_Memcpy_INTEL_fn)(cl_command_వరుస వరుస,
                                                      cl_bool blocking,
                                                      శూన్య * dst_ptr,
                                                      స్థిరరాశి శూన్య * src_ptr,
                                                      size_t కొలత,
                                                      cl_uint అంకె_events_లోపల_wait_list,
                                                      స్థిరరాశి cl_event * event_wait_list,
                                                      cl_event * event);

// Get the clEnqueueMemFillINTEL function pointer (returns nullptr if not available)
cl_వరుసలో_చేర్చు_Mem_Fill_INTEL_fn ggml_openvino_get_cl_వరుసలో_చేర్చు_Mem_Fill_INTEL();

// Get the clEnqueueMemcpyINTEL function pointer (returns nullptr if not available)
cl_వరుసలో_చేర్చు_Memcpy_INTEL_fn ggml_openvino_get_cl_వరుసలో_చేర్చు_Memcpy_INTEL();

// =====================================================
// Global Device Configuration (singleton)
// =====================================================
// Initialized once during backend init from GGML_OPENVINO_DEVICE env var

నిర్మాణం ggml_openvino_device_config {
    std::string device_name = "CPU";
    std::సదిశ<std::string> అందుబాటు_devices;
    bool is_npu = false;
    bool initialized = false;
    std::optional<ov::RemoteContext> remote_context;
    size_t max_alloc_కొలత = SIZE_MAX;
    ov::AnyMap compile_config;
    std::unordered_map<std::string, std::string> environment_variables;
    cl_command_వరుస cl_వరుస = nullptr;
    cl_వరుసలో_చేర్చు_Mem_Fill_INTEL_fn cl_mem_fill_fn = nullptr;
    cl_వరుసలో_చేర్చు_Memcpy_INTEL_fn cl_mem_cpy_fn = nullptr;

    శూన్య init();
    ~ggml_openvino_device_config();
};

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config();

// Initialize device config (call during backend init)
శూన్య ggml_openvino_init_device_config();

// Get the device name
స్థిరరాశి std::string & ggml_openvino_get_device_name();

// Get all available physical OpenVINO devices
std::సదిశ<std::string> ggml_openvino_get_అందుబాటు_devices();

// Human-readable device name, e.g. "Intel(R) AI Boost (NPU 4000)"; the device id if unavailable
std::string ggml_openvino_get_device_description(స్థిరరాశి std::string & device_name);

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
స్థిరరాశి అక్షరం * ggml_openvino_getenv_str(స్థిరరాశి అక్షరం * var, స్థిరరాశి అక్షరం * default_విలువ = nullptr);
పూర్ణాంకం ggml_openvino_getenv_int(స్థిరరాశి అక్షరం * var, పూర్ణాంకం default_విలువ = 0);

// Memory optimization toggles. GGML_OPENVINO_MEMORY_OPTIMIZE is an umbrella
// switch; the fine-grained env vars still override it when explicitly set.
bool ggml_openvino_reduce_compile_mem_enabled();
bool ggml_openvino_release_weights_enabled();

// Check if running on NPU
bool ggml_openvino_is_npu();

// Check if running on a GPU (GPU, GPU.0, GPU.1, ...)
bool ggml_openvino_is_gpu();

// Largest single memory object the device can allocate, SIZE_MAX when there is no known limit
size_t ggml_openvino_max_alloc_కొలత();

// Host weight-buffer release (GGML_OPENVINO_RELEASE_WEIGHTS, GPU only).
// register: record a host weight buffer (idempotent per data pointer).
// release:  madvise(MADV_DONTNEED) all registered buffers, dropping their RSS.
// released: true once release has run (used to fail-fast on post-release recompile).
శూన్య ggml_openvino_రిజిస్టరు_బరువు_బఫరు(శూన్య * దత్తాంశం, size_t కొలత);
శూన్య ggml_openvino_release_బరువు_buffers();
bool ggml_openvino_బరువు_buffers_released();

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(స్థిరరాశి ggml_tensor * tensor, bool no_requant = false);

// =====================================================
// OpenVINO Tensor Extra Types
// =====================================================
// These types are stored in tensor->extra by the OpenVINO backend buffer.
// They allow:
// 1. Pre-built ov::Constant nodes for weights (avoiding memcpy during graph construction)
// 2. ov::Tensor wrappers for KV cache / compute tensors (for direct use with infer_request)

// Base class for OpenVINO tensor extra data
నిర్మాణం ggml_openvino_extra_పునాది {
    గణన class Type { బరువు_2, QUANTIZED_బరువు, TENSOR };
    Type type;
    మాయా ~ggml_openvino_extra_పునాది() = అప్రమేయ;
రక్షిత:
    explicit ggml_openvino_extra_పునాది(Type t) : type(t) {}
};

// Extra data for F16/F32/BF16 weight tensors - stores the pre-built weight node
నిర్మాణం ggml_openvino_బరువు_extra : బహిరంగ ggml_openvino_extra_పునాది {
    ov::Tensor weights;                     // The underlying weight data tensor
    std::shared_ptr<ov::కణుపు_2> బరువు_కణుపు;  // Pre-built OpenVINO weight node

    ggml_openvino_బరువు_extra(ov::Tensor w, std::shared_ptr<ov::కణుపు_2> n) :
        ggml_openvino_extra_పునాది(Type::బరువు_2),
        weights(std::move(w)),
        బరువు_కణుపు(std::move(n)) {}
};

// Extra data for quantized weight tensors - stores extracted weights/scales/zp and weight node
నిర్మాణం ggml_openvino_quantized_బరువు_extra : బహిరంగ ggml_openvino_extra_పునాది {
    ov::Tensor weights;                     // U4 or U8 extracted weights
    ov::Tensor scales;                      // F16 scales
    ov::Tensor zp;                          // U4 or U8 zero points (same type as weights)
    std::shared_ptr<ov::కణుపు_2> బరువు_కణుపు;  // Pre-built OpenVINO weight subgraph

    ggml_openvino_quantized_బరువు_extra(ov::Tensor w, ov::Tensor s, ov::Tensor z, std::shared_ptr<ov::కణుపు_2> n) :
        ggml_openvino_extra_పునాది(Type::QUANTIZED_బరువు),
        weights(std::move(w)),
        scales(std::move(s)),
        zp(std::move(z)),
        బరువు_కణుపు(std::move(n)) {}
};

// Extra data for KV cache / compute tensors - stores ov::Tensor for infer_request
నిర్మాణం ggml_openvino_tensor_extra : బహిరంగ ggml_openvino_extra_పునాది {
    std::shared_ptr<ov::Tensor> tensor;  // For direct use with infer_request

    explicit ggml_openvino_tensor_extra(std::shared_ptr<ov::Tensor> t) :
        ggml_openvino_extra_పునాది(Type::TENSOR),
        tensor(std::move(t)) {}
};

// =====================================================
// Extracted Size Calculation for Quantized Tensors
// =====================================================
// For quantized tensors, we need extra space to store extracted weights, scales, and zero points.
// Returns the total size needed in the buffer for extracted data.

నిర్మాణం ggml_openvino_extracted_layout {
    size_t మొత్తం_కొలత = 0;      // Total bytes needed
    size_t weights_స్థానభ్రంశం = 0;  // Offset to weights in buffer
    size_t weights_కొలత = 0;    // Size of weights in bytes
    size_t scales_స్థానభ్రంశం = 0;   // Offset to scales in buffer
    size_t scales_కొలత = 0;     // Size of scales in bytes
    size_t zp_స్థానభ్రంశం = 0;       // Offset to zero points in buffer
    size_t zp_కొలత = 0;         // Size of zero points in bytes (U4 or U8)
    bool is_u4;                 // true for U4 weights, false for U8
    int64_t weights_చొప్పున_ఖండం;  // weights per scale/zp block
    bool is_సౌష్ఠవ;          // true for symmetric quantization

    // Requantization info
    bool is_requant = false;                     // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;  // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(స్థిరరాశి ggml_tensor * tensor, bool use_bias = false);

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(స్థిరరాశి ggml_tensor * tensor, bool is_remote);

// Check if a tensor's buffer uses remote (device) memory (e.g. GPU USM)
bool ggml_openvino_బఫరు_is_remote(స్థిరరాశి ggml_tensor * tensor);

// Register an extra with the tensor's OpenVINO buffer context for proper lifetime management.
// This sets tensor->extra and tracks the extra in the buffer context for cleanup.
శూన్య ggml_openvino_బఫరు_రిజిస్టరు_extra(ggml_tensor * tensor, ggml_openvino_extra_పునాది * extra);

// =====================================================
// OpenVINO Backend Context and Interface
// =====================================================
నిర్మాణం ggml_backend_openvino_context {
    పూర్ణాంకం device = 0;
    std::string name = "OpenVINO";
    std::string description = "OpenVINO Backend Context";

    std::shared_ptr<శూన్య> runtime_context = nullptr;

    ggml_backend_openvino_context() = అప్రమేయ;
};
