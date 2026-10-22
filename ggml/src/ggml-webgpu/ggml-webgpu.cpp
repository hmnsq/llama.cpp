/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-webgpu.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-webgpu-shader-lib.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <webgpu/webgpu_cpp.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ compute_2d_workgroups(uint32_t ਕੁੱਲ_wg, uint32_t max_ਪ੍ਰਤੀ_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(ਕੁੱਲ_wg, max_ਪ੍ਰਤੀ_dim));
    wg_x = CEIL_DIV(ਕੁੱਲ_wg, wg_y);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t ggml_webgpu_u32_from_f32(ਦਸ਼ਮਲਵ value) {
    uint32_t ਬਿੱਟ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਿੱਟ, &value, ਆਕਾਰ(ਬਿੱਟ));
    ਵਾਪਸ ਬਿੱਟ;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(ਸੁਨੇਹਾ_2)  std::ਬਾਹਰ << ਸੁਨੇਹਾ_2 << std::ਸਤਰ_ਅੰਤ
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#ਨਹੀਂਤਾਂ
#    define WEBGPU_LOG_DEBUG(ਸੁਨੇਹਾ_2) ((ਖਾਲੀ) 0)
#ਜੇ_ਅੰਤ  // GGML_WEBGPU_DEBUG

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(id) ਸਵੈਚਾਲਿਤ cpu_ਕੁੱਲ_ਸ਼ੁਰੂ##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(id, ctx)                                                         \
        ਸਵੈਚਾਲਿਤ   cpu_ਕੁੱਲ_ਸਿਰਾ##id = std::chrono::high_resolution_clock::now();                            \
        ਦੁੱਗਣਾ cpu_ਕੁੱਲ_ਸਮਾਂ##id =                                                                      \
            std::chrono::duration<ਦੁੱਗਣਾ, std::milli>(cpu_ਕੁੱਲ_ਸਿਰਾ##id - cpu_ਕੁੱਲ_ਸ਼ੁਰੂ##id).count(); \
        (ctx)->cpu_ਸਮਾਂ_ms[#id] += cpu_ਕੁੱਲ_ਸਮਾਂ##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_ਸ਼ੁਰੂ(id) ਸਵੈਚਾਲਿਤ cpu_detail_ਸ਼ੁਰੂ##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_ਸਿਰਾ(id, ctx)                                                          \
        ਸਵੈਚਾਲਿਤ   cpu_detail_ਸਿਰਾ##id = std::chrono::high_resolution_clock::now();                             \
        ਦੁੱਗਣਾ cpu_detail_ਸਮਾਂ##id =                                                                       \
            std::chrono::duration<ਦੁੱਗਣਾ, std::milli>(cpu_detail_ਸਿਰਾ##id - cpu_detail_ਸ਼ੁਰੂ##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_ਸਮਾਂ##id;
#ਨਹੀਂਤਾਂ
#    define WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(id)
#    define WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_ਸ਼ੁਰੂ(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_ਸਿਰਾ(id, ctx)
#ਜੇ_ਅੰਤ  // GGML_WEBGPU_CPU_PROFILE

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_ਗਿਣਤੀ        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_ਮਾਪ_ਬਾਈਟਾਂ (WEBGPU_MAX_PROFILE_QUERY_ਗਿਣਤੀ * ਆਕਾਰ(uint64_t))
#ਜੇ_ਅੰਤ

/* Constants */

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_ਮਾਪ 64u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_ਅੰਕ_PARAM_ਥਾਂ_SAFETY_MARGIN      10u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_PARAMS_BUF_ਮਾਪ_ਬਾਈਟਾਂ             128  // enough for 32 parameters
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ERROR_BUF_ਮਾਪ_ਬਾਈਟਾਂ     4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
ਸਥਿਰ ਖਾਲੀ * ਅਚਲ webgpu_ptr_ਅਧਾਰ = (ਖਾਲੀ *) (uintptr_t) 0x1000;  // NOLINT

ਸਥਿਰ size_t ggml_webgpu_tensor_ਵਿੱਥ(ਅਚਲ ggml_tensor * tensor) {
    ਅਚਲ ggml_tensor * ਅਧਾਰ_tensor = tensor->view_src ? tensor->view_src : tensor;
    ਵਾਪਸ (size_t) ((uintptr_t) ਅਧਾਰ_tensor->ਅੰਕੜੇ - (uintptr_t) webgpu_ptr_ਅਧਾਰ) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
ਸਥਿਰ ਖਾਲੀ ggml_webgpu_create_ਬਫਰ(wgpu::Device &    device,
                                      wgpu::ਬਫਰ_2 &    ਬਫਰ,
                                      size_t            size,
                                      wgpu::ਬਫਰ_Usage usage,
                                      ਅਚਲ ਅੱਖਰ *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
ਬਣਤਰ webgpu_param_arena {
    wgpu::ਬਫਰ_2 ਬਫਰ;
    size_t       ਥਾਂ_stride = 0;
    size_t       ਥਾਂ_ਮਾਪ   = 0;
    uint32_t     ਥਾਂ_ਗਿਣਤੀ  = 0;
    uint32_t     ਅਗਲਾ_ਥਾਂ   = 0;

    ਖਾਲੀ init(wgpu::Device device, size_t ਥਾਂ_ਮਾਪ, uint32_t ਥਾਂ_ਗਿਣਤੀ, size_t alignment) {
        ਇਹ->ਥਾਂ_stride = ROUNDUP_POW2(ਥਾਂ_ਮਾਪ, alignment);
        ਇਹ->ਥਾਂ_ਮਾਪ   = ਥਾਂ_ਮਾਪ;
        ਇਹ->ਥਾਂ_ਗਿਣਤੀ  = ਥਾਂ_ਗਿਣਤੀ;
        ਇਹ->ਅਗਲਾ_ਥਾਂ   = 0;

        ggml_webgpu_create_ਬਫਰ(device, ਬਫਰ, ਇਹ->ਥਾਂ_stride * ਥਾਂ_ਗਿਣਤੀ,
                                  wgpu::ਬਫਰ_Usage::CopyDst | wgpu::ਬਫਰ_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_ਥਾਂ(size_t size) {
        GGML_ASSERT(size <= ਥਾਂ_ਮਾਪ);
        ਜੇ (ਅਗਲਾ_ਥਾਂ >= ਥਾਂ_ਗਿਣਤੀ) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        ਵਾਪਸ ਥਾਂ_stride * ਅਗਲਾ_ਥਾਂ++;
    }

    ਖਾਲੀ reset() { ਅਗਲਾ_ਥਾਂ = 0; }

    ਖਾਲੀ cleanup() {
        ਜੇ (ਬਫਰ) {
            ਬਫਰ.Destroy();
            ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    ~webgpu_param_arena() { ਇਹ->cleanup(); }
};

ਬਣਤਰ webgpu_encoded_op {
    uint32_t ਅੰਕ_kernels = 0;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_ਨਾਂ;
#ਜੇ_ਅੰਤ
};

ਬਣਤਰ webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> bind_ਸਮੂਹ_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

ਬਣਤਰ webgpu_capabilities {
    wgpu::Limits limits;
    ਤਾਰਕਿਕ         supports_subgroups       = ਝੂਠ;
    ਤਾਰਕਿਕ         supports_subgroup_ਮੈਟ੍ਰਿਕਸ = ਝੂਠ;
    ਤਾਰਕਿਕ         supports_dot_ਗੁਣਨਫਲ     = ਝੂਠ;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_ਮਾਪ     = 0;
    uint32_t min_subgroup_ਮਾਪ = 0;
    uint32_t max_subgroup_ਮਾਪ = 0;
    size_t   memset_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
};

// Stores global webgpu members
ਬਣਤਰ webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::ਕਤਾਰ_2    queue;
    uint32_t       command_submit_batch_ਮਾਪ = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_ਮਾਪ;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::ਬਫਰ_2         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::ਬਫਰ_2    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, ਦੁੱਗਣਾ> cpu_ਸਮਾਂ_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, ਦੁੱਗਣਾ> cpu_detail_ms;
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_DEBUG
    wgpu::ਬਫਰ_2 debug_host_buf;
    wgpu::ਬਫਰ_2 debug_dev_buf;
#ਜੇ_ਅੰਤ

    ~webgpu_global_context_struct() {
        ਜੇ (ਇਹ->get_tensor_staging_buf) {
            ਇਹ->get_tensor_staging_buf.Destroy();
            ਇਹ->get_tensor_staging_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
        ਜੇ (ਇਹ->memset_params_buf) {
            ਇਹ->memset_params_buf.Destroy();
            ਇਹ->memset_params_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_DEBUG
        ਜੇ (ਇਹ->debug_host_buf) {
            ਇਹ->debug_host_buf.Destroy();
            ਇਹ->debug_host_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
        ਜੇ (ਇਹ->debug_dev_buf) {
            ਇਹ->debug_dev_buf.Destroy();
            ਇਹ->debug_dev_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
#ਜੇ_ਅੰਤ
    }
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
ਬਣਤਰ webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::ਬਫਰ_2             ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf;
    wgpu::ਬਫਰ_2             ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf;
    wgpu::CommandEncoder     ਸਰਗਰਮ_command_encoder;
    wgpu::ComputePassEncoder ਸਰਗਰਮ_compute_pass;
    ਤਾਰਕਿਕ                     batch_compute_passes = ਸੱਚ;

    size_t memset_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, ਦੁੱਗਣਾ> shader_gpu_ਸਮਾਂ_ms;
    wgpu::ਬਫਰ_2                            profile_timestamp_dev_buf;
    wgpu::ਬਫਰ_2                            profile_timestamp_host_buf;
    wgpu::Query_ਸੈੱਟ                          profile_timestamp_query_ਸੈੱਟ;
    uint32_t                                profile_timestamp_query_ਗਿਣਤੀ = 0;
#ਜੇ_ਅੰਤ

    ~webgpu_context_struct() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
        ਜੇ (ਇਹ->profile_timestamp_host_buf) {
            ਇਹ->profile_timestamp_host_buf.Destroy();
            ਇਹ->profile_timestamp_host_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
        ਜੇ (ਇਹ->profile_timestamp_dev_buf) {
            ਇਹ->profile_timestamp_dev_buf.Destroy();
            ਇਹ->profile_timestamp_dev_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
        ਜੇ (ਇਹ->profile_timestamp_query_ਸੈੱਟ) {
            ਇਹ->profile_timestamp_query_ਸੈੱਟ.Destroy();
            ਇਹ->profile_timestamp_query_ਸੈੱਟ = ਖਾਲੀ_ਸੂਚਕ;
        }
#ਜੇ_ਅੰਤ
        ਜੇ (ਇਹ->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf) {
            ਇਹ->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf.Destroy();
            ਇਹ->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
        ਜੇ (ਇਹ->ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf) {
            ਇਹ->ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf.Destroy();
            ਇਹ->ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf = ਖਾਲੀ_ਸੂਚਕ;
        }
    }
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
ਬਣਤਰ ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_ਗਿਣਤੀ;
    ਅਚਲ ਅੱਖਰ *          name;
};

// Per-device struct for the global logical device interface
ਬਣਤਰ ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
ਬਣਤਰ ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
ਬਣਤਰ ggml_backend_webgpu_ਬਫਰ_context {
    wgpu::ਬਫਰ_2          ਬਫਰ;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_ਬਫਰ_context(wgpu::ਬਫਰ_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        ਬਫਰ(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

ਸਥਿਰ webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ਅਚਲ ਅੱਖਰ *                             shader_ਕੋਡ,
                                                   ਅਚਲ ਅੱਖਰ *                             label,
                                                   ਅਚਲ std::vector<wgpu::Constant_ਇੰਦਰਾਜ> & constants = {}) {
    wgpu::Shader_ਸਰੋਤ_WGSL shader_ਸਰੋਤ;
    shader_ਸਰੋਤ.code = shader_ਕੋਡ;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.ਅਗਲਾ_ਵਿੱਚ_Chain = &shader_ਸਰੋਤ;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.ਇੰਦਰਾਜ_ਬਿੰਦੂ_2 = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = ਖਾਲੀ_ਸੂਚਕ;  // nullptr means auto layout
    ਜੇ (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.ਅੰਕੜੇ();
        pipeline_desc.compute.constant_ਗਿਣਤੀ = constants.size();
    }
    ਵਾਪਸ { device.CreateComputePipeline(&pipeline_desc), label };
}

ਸਥਿਰ ਖਾਲੀ ggml_webgpu_create_ਬਫਰ(wgpu::Device &    device,
                                      wgpu::ਬਫਰ_2 &    ਬਫਰ,
                                      size_t            size,
                                      wgpu::ਬਫਰ_Usage usage,
                                      ਅਚਲ ਅੱਖਰ *      label) {
    wgpu::ਬਫਰ_Descriptor ਬਫਰ_desc;
    ਬਫਰ_desc.size             = size;
    ਬਫਰ_desc.usage            = usage;
    ਬਫਰ_desc.label            = label;
    ਬਫਰ_desc.mappedAtCreation = ਝੂਠ;

    // TODO: error handling
    ਬਫਰ = device.Create_ਬਫਰ(&ਬਫਰ_desc);
}

ਸਥਿਰ wgpu::ਬਫਰ_2 ggml_webgpu_tensor_buf(ਅਚਲ ggml_tensor * tensor) {
    ggml_backend_webgpu_ਬਫਰ_context * ctx = (ggml_backend_webgpu_ਬਫਰ_context *) tensor->ਬਫਰ->context;
    ਵਾਪਸ ctx->ਬਫਰ;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
ਸਥਿਰ size_t ggml_webgpu_tensor_align_ਵਿੱਥ(ਅਚਲ ggml_tensor * t, size_t alignment) {
    ਅਚਲ size_t ਵਿੱਥ    = ggml_webgpu_tensor_ਵਿੱਥ(t);
    ਅਚਲ size_t type_ਮਾਪ = ggml_type_ਮਾਪ(t->type);
    size_t       aligned   = ਵਿੱਥ & ~(alignment - 1);
    ਜਦੋਂ ((ਵਿੱਥ - aligned) % type_ਮਾਪ != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    ਵਾਪਸ aligned;
}

ਸਥਿਰ size_t ggml_webgpu_tensor_misalignment(ਅਚਲ ggml_tensor * t, size_t alignment) {
    ਵਾਪਸ ggml_webgpu_tensor_ਵਿੱਥ(t) - ggml_webgpu_tensor_align_ਵਿੱਥ(t, alignment);
}

ਸਥਿਰ size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ਅਚਲ ggml_tensor * t) {
    ਵਾਪਸ ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
}

ਸਥਿਰ size_t ggml_webgpu_tensor_align_ਵਿੱਥ(webgpu_context & ctx, ਅਚਲ ggml_tensor * t) {
    ਵਾਪਸ ggml_webgpu_tensor_align_ਵਿੱਥ(t, ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
}

ਸਥਿਰ size_t ggml_webgpu_tensor_binding_ਮਾਪ(ਅਚਲ ggml_tensor * t, size_t alignment) {
    ਵਾਪਸ ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

ਸਥਿਰ size_t ggml_webgpu_tensor_binding_ਮਾਪ(webgpu_context & ctx, ਅਚਲ ggml_tensor * t) {
    ਵਾਪਸ ggml_webgpu_tensor_binding_ਮਾਪ(t, ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_webgpu_tensor_binding_overlap(ਅਚਲ webgpu_global_context & global_ctx,
                                               ਅਚਲ ggml_tensor *           a,
                                               ਅਚਲ ggml_tensor *           b) {
    ਜੇ (a->ਬਫਰ != b->ਬਫਰ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ size_t alignment = global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;
    ਅਚਲ size_t a_ਵਿੱਥ  = ggml_webgpu_tensor_align_ਵਿੱਥ(a, alignment);
    ਅਚਲ size_t b_ਵਿੱਥ  = ggml_webgpu_tensor_align_ਵਿੱਥ(b, alignment);
    ਵਾਪਸ a_ਵਿੱਥ < b_ਵਿੱਥ + ggml_webgpu_tensor_binding_ਮਾਪ(b, alignment) &&
           b_ਵਿੱਥ < a_ਵਿੱਥ + ggml_webgpu_tensor_binding_ਮਾਪ(a, alignment);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_webgpu_tensor_binding_overlap_range(ਅਚਲ webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_ਬਫਰ_t         ਬਫਰ,
                                                     size_t                        ਵਿੱਥ,
                                                     size_t                        size) {
    ਜੇ (tensor->ਬਫਰ != ਬਫਰ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ size_t alignment     = global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;
    ਅਚਲ size_t tensor_ਵਿੱਥ = ggml_webgpu_tensor_align_ਵਿੱਥ(tensor, alignment);
    ਵਾਪਸ tensor_ਵਿੱਥ < ਵਿੱਥ + size && ਵਿੱਥ < tensor_ਵਿੱਥ + ggml_webgpu_tensor_binding_ਮਾਪ(tensor, alignment);
}

ਬਣਤਰ ggml_webgpu_merged_binding_range {
    size_t ਵਿੱਥ;
    size_t size;
};

ਸਥਿਰ ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_ਵਿੱਥ = SIZE_MAX;
    size_t merged_ਸਿਰਾ    = 0;

    ਲਈ (ggml_tensor * tensor : tensors) {
        ਅਚਲ size_t bind_ਵਿੱਥ = ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, tensor);
        ਅਚਲ size_t bind_ਸਿਰਾ    = bind_ਵਿੱਥ + ggml_webgpu_tensor_binding_ਮਾਪ(ctx, tensor);

        merged_ਵਿੱਥ = std::min(merged_ਵਿੱਥ, bind_ਵਿੱਥ);
        merged_ਸਿਰਾ    = std::max(merged_ਸਿਰਾ, bind_ਸਿਰਾ);
    }

    ਵਾਪਸ { merged_ਵਿੱਥ, merged_ਸਿਰਾ - merged_ਵਿੱਥ };
}

ਸਥਿਰ uint32_t ggml_webgpu_tensor_merged_element_ਵਿੱਥ(ਅਚਲ ggml_tensor *                      tensor,
                                                         ਅਚਲ ggml_webgpu_merged_binding_range & merged_range) {
    ਵਾਪਸ (uint32_t) ((ggml_webgpu_tensor_ਵਿੱਥ(tensor) - merged_range.ਵਿੱਥ) / ggml_type_ਮਾਪ(tensor->type));
}

ਸਥਿਰ wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(uint32_t     binding,
                                                              wgpu::ਬਫਰ_2 ਬਫਰ,
                                                              uint64_t     ਵਿੱਥ,
                                                              uint64_t     size) {
    wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ ਇੰਦਰਾਜ = {};
    ਇੰਦਰਾਜ.binding              = binding;
    ਇੰਦਰਾਜ.ਬਫਰ               = std::move(ਬਫਰ);
    ਇੰਦਰਾਜ.ਵਿੱਥ               = ਵਿੱਥ;
    ਇੰਦਰਾਜ.size                 = size;
    ਵਾਪਸ ਇੰਦਰਾਜ;
}

ਸਥਿਰ wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    ਵਾਪਸ ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, tensor),
                                             ggml_webgpu_tensor_binding_ਮਾਪ(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_check_wait_ਸਥਿਤੀ(wgpu::Wait_ਸਥਿਤੀ wait_ਸਥਿਤੀ,
                                                  T                callback_ਸਥਿਤੀ,
                                                  T                success_ਸਥਿਤੀ,
                                                  ਅਚਲ ਅੱਖਰ *     wait_name,
                                                  ਅਚਲ ਅੱਖਰ *     failure_name,
                                                  ਅਚਲ ਅੱਖਰ *     callback_ਸੁਨੇਹਾ) {
    ਜੇ (wait_ਸਥਿਤੀ == wgpu::Wait_ਸਥਿਤੀ::Timed_ਬਾਹਰਲਾ) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    ਜੇ (wait_ਸਥਿਤੀ == wgpu::Wait_ਸਥਿਤੀ::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    ਜੇ (callback_ਸਥਿਤੀ != success_ਸਥਿਤੀ) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<ਪੂਰਨ>(callback_ਸਥਿਤੀ),
                   callback_ਸੁਨੇਹਾ);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
ਸਥਿਰ uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    ਵਾਪਸ UINT32_MAX;
}

ਸਥਿਰ uint32_t ggml_backend_webgpu_get_command_submit_batch_ਮਾਪ() {
    ਵਾਪਸ WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_ਮਾਪ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_wait_ਕਤਾਰ(webgpu_global_context & ctx) {
    wgpu::ਕਤਾਰ_Work_ਪੂਰਾ_ਸਥਿਤੀ callback_ਸਥਿਤੀ = wgpu::ਕਤਾਰ_Work_ਪੂਰਾ_ਸਥਿਤੀ::Error;
    std::string               callback_ਸੁਨੇਹਾ;

    ਅਚਲ wgpu::Wait_ਸਥਿਤੀ wait_ਸਥਿਤੀ = ctx->instance.WaitAny(
        ctx->queue.On_Submitted_Work_ਪੂਰਾ(
            wgpu::Callback_ਬਹੁਲਕ::AllowSpontaneous,
            [&callback_ਸਥਿਤੀ, &callback_ਸੁਨੇਹਾ](wgpu::ਕਤਾਰ_Work_ਪੂਰਾ_ਸਥਿਤੀ ਸਥਿਤੀ, wgpu::StringView ਸੁਨੇਹਾ) {
                callback_ਸਥਿਤੀ  = ਸਥਿਤੀ;
                callback_ਸੁਨੇਹਾ = std::string(ਸੁਨੇਹਾ);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_ਸਥਿਤੀ(wait_ਸਥਿਤੀ, callback_ਸਥਿਤੀ, wgpu::ਕਤਾਰ_Work_ਪੂਰਾ_ਸਥਿਤੀ::Success,
                                          "Queue wait", "Queue work", callback_ਸੁਨੇਹਾ.c_str());
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_map_ਬਫਰ(webgpu_global_context & ctx,
                                           wgpu::ਬਫਰ_2 &          ਬਫਰ,
                                           wgpu::Map_ਬਹੁਲਕ           mode,
                                           size_t                  ਵਿੱਥ,
                                           size_t                  size) {
    wgpu::Map_Async_ਸਥਿਤੀ callback_ਸਥਿਤੀ = wgpu::Map_Async_ਸਥਿਤੀ::Error;
    std::string          callback_ਸੁਨੇਹਾ;

    ਅਚਲ wgpu::Wait_ਸਥਿਤੀ wait_ਸਥਿਤੀ = ctx->instance.WaitAny(
        ਬਫਰ.MapAsync(mode, ਵਿੱਥ, size, wgpu::Callback_ਬਹੁਲਕ::AllowSpontaneous,
                        [&callback_ਸਥਿਤੀ, &callback_ਸੁਨੇਹਾ](wgpu::Map_Async_ਸਥਿਤੀ ਸਥਿਤੀ, wgpu::StringView ਸੁਨੇਹਾ) {
                            callback_ਸਥਿਤੀ  = ਸਥਿਤੀ;
                            callback_ਸੁਨੇਹਾ = std::string(ਸੁਨੇਹਾ);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_ਸਥਿਤੀ(wait_ਸਥਿਤੀ, callback_ਸਥਿਤੀ, wgpu::Map_Async_ਸਥਿਤੀ::Success,
                                          "Buffer map wait", "Buffer map", callback_ਸੁਨੇਹਾ.c_str());
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ਅਚਲ wgpu::Command_ਬਫਰ commands,
                                                uint32_t &                ਅੰਕ_inflight_batches) {
    ਜੇ (ਅੰਕ_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_ਕਤਾਰ(ctx->global_ctx);
        ਅੰਕ_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    ਅੰਕ_inflight_batches++;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_ਬਫਰ_ਮੰਜ਼ਿਲ_ਬਫਰ(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_ਮਾਪ());
    wgpu::Command_ਬਫਰ commands = encoder.ਅੰਤ_2();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_ਬਫਰ(ctx, ctx->debug_host_buf, wgpu::Map_ਬਹੁਲਕ::Read, 0, ctx->debug_host_buf.Get_ਮਾਪ());
    ਅਚਲ ਦਸ਼ਮਲਵ * debug_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ctx->debug_host_buf.GetConstMappedRange();
    std::ਬਾਹਰ << "debug[0]: " << debug_ਅੰਕੜੇ[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ਅਚਲ std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            ਨਤੀਜਾ = {};
    std::vector<wgpu::Bind_ਸਮੂਹ> bind_groups;
    std::vector<size_t>          param_offsets;
    ਨਤੀਜਾ.ਅੰਕ_kernels = dispatches.size();

    ਲਈ (size_t i = 0; i < dispatches.size(); i++) {
        ਅਚਲ webgpu_dispatch_desc & dispatch     = dispatches[i];
        ਅਚਲ size_t                 param_ਮਾਪ   = dispatch.params.size() * ਆਕਾਰ(uint32_t);
        ਅਚਲ size_t                 param_ਵਿੱਥ = ctx->param_arena.alloc_ਥਾਂ(param_ਮਾਪ);

        std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries            = dispatch.bind_ਸਮੂਹ_entries;
        uint32_t                          params_binding_ਅੰਕ = entries.size();
        entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(params_binding_ਅੰਕ, ctx->param_arena.ਬਫਰ, param_ਵਿੱਥ,
                                                            ctx->param_arena.ਥਾਂ_ਮਾਪ));

        wgpu::Bind_ਸਮੂਹ_Descriptor bind_ਸਮੂਹ_desc;
        bind_ਸਮੂਹ_desc.layout     = dispatch.pipeline.pipeline.Get_Bind_ਸਮੂਹ_Layout(0);
        bind_ਸਮੂਹ_desc.ਇੰਦਰਾਜ_ਗਿਣਤੀ = entries.size();
        bind_ਸਮੂਹ_desc.entries    = entries.ਅੰਕੜੇ();
        bind_ਸਮੂਹ_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.Create_Bind_ਸਮੂਹ(&bind_ਸਮੂਹ_desc));
        param_offsets.push_back(param_ਵਿੱਥ);
    }

    ਲਈ (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_ਬਫਰ(ctx->param_arena.ਬਫਰ, param_offsets[i], dispatches[i].params.ਅੰਕੜੇ(),
                                           dispatches[i].params.size() * ਆਕਾਰ(uint32_t));
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    ਲਈ (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_ਗਿਣਤੀ + 2 <= WEBGPU_MAX_PROFILE_QUERY_ਗਿਣਤੀ);
        ਅਚਲ uint32_t query_begin = ctx->profile_timestamp_query_ਗਿਣਤੀ++;
        ਅਚਲ uint32_t query_ਸਿਰਾ   = ctx->profile_timestamp_query_ਗਿਣਤੀ++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.query_ਸੈੱਟ                    = ctx->profile_timestamp_query_ਸੈੱਟ;
        ts_writes.beginning_ਦਾ_Pass_Write_ਸੂਚਕਾਂਕ   = query_begin;
        ts_writes.ਸਿਰਾ_ਦਾ_Pass_Write_ਸੂਚਕਾਂਕ         = query_ਸਿਰਾ;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->ਸਰਗਰਮ_command_encoder.BeginComputePass(&pass_desc);

        pass.ਸੈੱਟ_Pipeline(dispatches[i].pipeline.pipeline);
        pass.ਸੈੱਟ_Bind_ਸਮੂਹ(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.ਪਹਿਲਾ, dispatches[i].workgroups.second, 1);
        pass.ਸਿਰਾ_2();
        ਨਤੀਜਾ.pipeline_ਨਾਂ.push_back(dispatches[i].pipeline.name);
    }
#ਨਹੀਂਤਾਂ
    ਲਈ (size_t i = 0; i < dispatches.size(); i++) {
        ਜੇ (ctx->batch_compute_passes) {
            ctx->ਸਰਗਰਮ_compute_pass.ਸੈੱਟ_Pipeline(dispatches[i].pipeline.pipeline);
            ctx->ਸਰਗਰਮ_compute_pass.ਸੈੱਟ_Bind_ਸਮੂਹ(0, bind_groups[i]);
            ctx->ਸਰਗਰਮ_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.ਪਹਿਲਾ, dispatches[i].workgroups.second,
                                                        1);
        } ਨਹੀਂਤਾਂ {
            wgpu::ComputePassEncoder pass = ctx->ਸਰਗਰਮ_command_encoder.BeginComputePass();
            pass.ਸੈੱਟ_Pipeline(dispatches[i].pipeline.pipeline);
            pass.ਸੈੱਟ_Bind_ਸਮੂਹ(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.ਪਹਿਲਾ, dispatches[i].workgroups.second, 1);
            pass.ਸਿਰਾ_2();
        }
    }
#ਜੇ_ਅੰਤ

    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> bind_ਸਮੂਹ_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    ਵਾਪਸ ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_ਸਮੂਹ_entries), { wg_x, wg_y } },
    });
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_ਬਫਰ_memset(webgpu_global_context & ctx,
                                              wgpu::ਬਫਰ_2 &          buf,
                                              uint32_t                value,
                                              size_t                  ਵਿੱਥ,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) ਵਿੱਥ, (uint32_t) size, value };
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, buf, 0, buf.Get_ਮਾਪ()) };
    size_t                            ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_wg =
        ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup * ctx->capabilities.memset_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_wg);

    ctx->queue.Write_ਬਫਰ(ctx->memset_params_buf, 0, params.ਅੰਕੜੇ(), params.size() * ਆਕਾਰ(uint32_t));

    wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ params_ਇੰਦਰਾਜ = {};
    params_ਇੰਦਰਾਜ.binding              = 1;
    params_ਇੰਦਰਾਜ.ਬਫਰ               = ctx->memset_params_buf;
    params_ਇੰਦਰਾਜ.ਵਿੱਥ               = 0;
    params_ਇੰਦਰਾਜ.size                 = WEBGPU_PARAMS_BUF_ਮਾਪ_ਬਾਈਟਾਂ;
    entries.push_back(params_ਇੰਦਰਾਜ);

    wgpu::Bind_ਸਮੂਹ_Descriptor bind_ਸਮੂਹ_desc;
    bind_ਸਮੂਹ_desc.layout     = ctx->memset_pipeline.pipeline.Get_Bind_ਸਮੂਹ_Layout(0);
    bind_ਸਮੂਹ_desc.ਇੰਦਰਾਜ_ਗਿਣਤੀ = entries.size();
    bind_ਸਮੂਹ_desc.entries    = entries.ਅੰਕੜੇ();
    bind_ਸਮੂਹ_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::Bind_ਸਮੂਹ bind_ਸਮੂਹ = ctx->device.Create_Bind_ਸਮੂਹ(&bind_ਸਮੂਹ_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.ਸੈੱਟ_Pipeline(ctx->memset_pipeline.pipeline);
    pass.ਸੈੱਟ_Bind_ਸਮੂਹ(0, bind_ਸਮੂਹ);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.ਸਿਰਾ_2();

    wgpu::Command_ਬਫਰ              command  = encoder.ਅੰਤ_2();
    std::vector<wgpu::Command_ਬਫਰ> commands = { command };
    ctx->queue.Submit(commands.size(), commands.ਅੰਕੜੇ());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_CPU_PROFILE
    std::ਬਾਹਰ << "\n[ggml_webgpu cpu profiling summary]\n";
    ਦੁੱਗਣਾ ਕੁੱਲ_cpu = 0.0;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kv : ctx->webgpu_ctx->global_ctx->cpu_ਸਮਾਂ_ms) {
        ਕੁੱਲ_cpu += kv.second;
    }
    std::ਬਾਹਰ << "ggml_webgpu: total cpu time: " << ਕੁੱਲ_cpu << " ms\n";
    std::ਬਾਹਰ << "ggml_webgpu: cpu breakdown:\n";
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kv : ctx->webgpu_ctx->global_ctx->cpu_ਸਮਾਂ_ms) {
        ਦੁੱਗਣਾ pct = (ਕੁੱਲ_cpu > 0.0) ? (kv.second / ਕੁੱਲ_cpu * 100.0) : 0.0;
        std::ਬਾਹਰ << "ggml_webgpu:  " << kv.ਪਹਿਲਾ << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    ਜੇ (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::ਬਾਹਰ << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        ਦੁੱਗਣਾ pct = (ਕੁੱਲ_cpu > 0.0) ? (kv.second / ਕੁੱਲ_cpu * 100.0) : 0.0;
        std::ਬਾਹਰ << "ggml_webgpu:  " << kv.ਪਹਿਲਾ << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    std::ਬਾਹਰ << "\n[ggml_webgpu gpu profiling summary]\n";
    ਦੁੱਗਣਾ ਕੁੱਲ_gpu = 0.0;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kv : ctx->webgpu_ctx->shader_gpu_ਸਮਾਂ_ms) {
        ਕੁੱਲ_gpu += kv.second;
    }
    std::ਬਾਹਰ << "ggml_webgpu: total gpu time (all shaders): " << ਕੁੱਲ_gpu << " ms\n";
    std::ਬਾਹਰ << "\nggml_webgpu: gpu breakdown:\n";
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kv : ctx->webgpu_ctx->shader_gpu_ਸਮਾਂ_ms) {
        ਦੁੱਗਣਾ pct = (ਕੁੱਲ_gpu > 0.0) ? (kv.second / ਕੁੱਲ_gpu * 100.0) : 0.0;
        std::ਬਾਹਰ << "ggml_webgpu:  " << kv.ਪਹਿਲਾ << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::ਬਾਹਰ << "ggml_webgpu: gpu/cpu ratio: " << (ਕੁੱਲ_cpu > 0.0 ? ਕੁੱਲ_gpu / ਕੁੱਲ_cpu : 0.0) << "\n";
#ਜੇ_ਅੰਤ

    ਮਿਟਾਓ ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_ਮਾਪ(src->type)), (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)), (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_ਮਾਪ(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ne, decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_ਸੈੱਟ(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ਸੈੱਟ_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ਅਚਲ ਤਾਰਕਿਕ inplace   = decisions->inplace;

    ਅਚਲ uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = (uint32_t) ggml_type_ਮਾਪ(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (((ਅਚਲ int32_t *) dst->op_params)[3] / dst_type_ਮਾਪ),

        (uint32_t) (src1->nb[0] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),

        1u,
        (uint32_t) (((ਅਚਲ int32_t *) dst->op_params)[0] / dst_type_ਮਾਪ),
        (uint32_t) (((ਅਚਲ int32_t *) dst->op_params)[1] / dst_type_ਮਾਪ),
        (uint32_t) (((ਅਚਲ int32_t *) dst->op_params)[2] / dst_type_ਮਾਪ),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries;
    uint32_t                          binding_ਸੂਚਕਾਂਕ = 0;
    ਜੇ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0));
        binding_ਸੂਚਕਾਂਕ++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਸੂਚਕਾਂਕ, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਸੂਚਕਾਂਕ + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ne, decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ਅਚਲ uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)),
        // Shapes
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        // Pad sizes
        (uint32_t) ggml_get_op_params_i32(dst, 0),
        (uint32_t) ggml_get_op_params_i32(dst, 1),
        (uint32_t) ggml_get_op_params_i32(dst, 2),
        (uint32_t) ggml_get_op_params_i32(dst, 3),
        (uint32_t) ggml_get_op_params_i32(dst, 4),
        (uint32_t) ggml_get_op_params_i32(dst, 5),
        (uint32_t) ggml_get_op_params_i32(dst, 6),
        (uint32_t) ggml_get_op_params_i32(dst, 7),
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ne, decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
    shader_lib_ctx.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_ਮਾਪ;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst),
    };

    ਅਚਲ uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_ਮਾਪ);
    ਅਚਲ uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    ਅਚਲ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ਅਚਲ int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ਅਚਲ int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ਅਚਲ int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],

        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
ਸਥਿਰ webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    ਅਚਲ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ਅਚਲ int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ਅਚਲ int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ਅਚਲ int32_t d1 = ggml_get_op_params_i32(dst, 5);

    // Scalar params matching conv2d_dw.wgsl (weight src0 [KW,KH,1,C], input src1, output dst).
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ਅਚਲ ਤਾਰਕਿਕ      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    ਅਚਲ int32_t s0    = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ int32_t s1    = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ int32_t p0    = ggml_get_op_params_i32(dst, 2);
    ਅਚਲ int32_t p1    = ggml_get_op_params_i32(dst, 3);
    ਅਚਲ int32_t d0    = ggml_get_op_params_i32(dst, 4);
    ਅਚਲ int32_t d1    = ggml_get_op_params_i32(dst, 5);
    ਅਚਲ ਤਾਰਕਿਕ    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    ਅਚਲ uint32_t KW = src0->ne[0];
    ਅਚਲ uint32_t KH = is_2D ? src0->ne[1] : 1;
    ਅਚਲ uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    ਅਚਲ uint32_t IW = src1->ne[0];
    ਅਚਲ uint32_t IH = is_2D ? src1->ne[1] : 1;
    ਅਚਲ uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    ਅਚਲ uint32_t OW = dst->ne[1];
    ਅਚਲ uint32_t OH = is_2D ? dst->ne[2] : 1;

    ਅਚਲ uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_ਮਾਪ(src1->type));
    ਅਚਲ uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)) : 0;
    ਅਚਲ uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type));
    ਅਚਲ uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type));

    ਅਚਲ uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_ਮਾਪ(dst->type));
    ਅਚਲ uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type));
    ਅਚਲ uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)) : 0;
    ਅਚਲ uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),

        si0,
        si1,
        si2,
        si3,
        so0,
        so1,
        so2,
        so3,

        KW,
        KH,
        IC,

        IW,
        IH,
        N,

        OW,
        OH,

        (uint32_t) s0,
        (uint32_t) s1,
        (uint32_t) p0,
        (uint32_t) p1,
        (uint32_t) d0,
        (uint32_t) d1,
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ਅਚਲ uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->ਟੋਕਨ_ਪ੍ਰਤੀ_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst),
    };

    ਅਚਲ uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->ਖੰਡ_ਮਾਪ_3);
    ਅਚਲ uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_ssm_ਜਾਂਚ(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    src3,
                                              ggml_tensor *    src4,
                                              ggml_tensor *    src5,
                                              ggml_tensor *    src6,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.src3                           = src3;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.src5                           = src5;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    ਜੇ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    ਜੇ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    ਜੇ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_ਜਾਂਚ_pipeline(shader_lib_ctx, xbc_overlap);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_ssm_ਜਾਂਚ_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t ਵਿੱਥ_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type));
    uint32_t ਵਿੱਥ_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ਮਾਪ(src2->type));
    uint32_t ਵਿੱਥ_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_ਮਾਪ(src3->type));
    uint32_t ਵਿੱਥ_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_ਮਾਪ(src4->type));
    uint32_t ਵਿੱਥ_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_ਮਾਪ(src5->type));
    uint32_t ਵਿੱਥ_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_ਮਾਪ(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    ਜੇ (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        ਵਿੱਥ_x         = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src1, xbc_merged_range);
        ਵਿੱਥ_B         = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src4, xbc_merged_range);
    } ਨਹੀਂਤਾਂ ਜੇ (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        ਵਿੱਥ_B         = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src4, xbc_merged_range);
        ਵਿੱਥ_C         = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src5, xbc_merged_range);
    } ਨਹੀਂਤਾਂ ਜੇ (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        ਵਿੱਥ_x         = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src1, xbc_merged_range);
        ਵਿੱਥ_B         = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src4, xbc_merged_range);
        ਵਿੱਥ_C         = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_ਵਿੱਥ = 0;
    size_t xbc_bind_ਮਾਪ   = 0;
    ਜੇ (xbc_overlap > 0) {
        xbc_bind_ਵਿੱਥ = xbc_merged_range.ਵਿੱਥ;
        xbc_bind_ਮਾਪ   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        ਵਿੱਥ_x,
        ਵਿੱਥ_dt,
        ਵਿੱਥ_A,
        ਵਿੱਥ_B,
        ਵਿੱਥ_C,
        ਵਿੱਥ_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_ਮਾਪ(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_ਮਾਪ(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_ਮਾਪ(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_ਮਾਪ(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_ਮਾਪ(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_ਮਾਪ(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_ਮਾਪ(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_ਮਾਪ(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_ਮਾਪ(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_ਅੰਕ = 0;

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, src0),
    };
    // xbc_merged binding
    ਜੇ (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(binding_ਅੰਕ++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_ਵਿੱਥ, xbc_bind_ਮਾਪ));
    }
    // x
    ਜੇ (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, src1));
    }
    // B
    ਜੇ (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, src4));
    }
    // C
    ਜੇ (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ++, dst));

    ਅਚਲ uint32_t ਕੁੱਲ_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ਅਚਲ uint32_t max_wg_ਪ੍ਰਤੀ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(ਕੁੱਲ_wg, max_wg_ਪ੍ਰਤੀ_dim, wg_x, wg_y);

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
                                                     ggml_tensor *    src0,
                                                     ggml_tensor *    src1,
                                                     ggml_tensor *    src2,
                                                     ggml_tensor *    src3,
                                                     ggml_tensor *    src4,
                                                     ggml_tensor *    src5,
                                                     ggml_tensor *    dst,
                                                     ggml_tensor *    dst_fuse) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.src3                           = src3;
    shader_lib_ctx.src4                           = src4;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.dst_fuse                       = dst_fuse;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ਅਚਲ uint32_t s_v      = (uint32_t) src2->ne[0];
    ਅਚਲ uint32_t h        = (uint32_t) src2->ne[1];
    ਅਚਲ uint32_t n_ਟੋਕਨ = (uint32_t) src2->ne[2];
    ਅਚਲ uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ਅਚਲ uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ਅਚਲ ਦਸ਼ਮਲਵ    scale    = 1.0f / sqrtf((ਦਸ਼ਮਲਵ) s_v);
    uint32_t       scale_u32;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale_u32, &scale, ਆਕਾਰ(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_ਟੋਕਨ,
        n_seqs,
        s_v * h * n_ਟੋਕਨ * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_ਮਾਪ(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_ਮਾਪ(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_ਮਾਪ(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_ਮਾਪ(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_ਮਾਪ(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_ਮਾਪ(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_ਮਾਪ(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_ਮਾਪ(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0), ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, src2), ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 4, src4), ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 6, dst),
    };

    ਜੇ (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 7, dst_fuse));
    }

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

ਸਥਿਰ std::optional<webgpu_encoded_op> ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(src) || ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(idx)) {
        ਵਾਪਸ std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ਸੈੱਟ_ਪੰਕਤੀਆਂ_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_ਮਾਪ(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)), (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)), (uint32_t) (idx->nb[0] / ggml_type_ਮਾਪ(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_ਮਾਪ(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_ਮਾਪ(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst));

    ਜੇ (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(3, ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf, 0,
                                                            ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf.Get_ਮਾਪ()));
    }

    uint32_t ਧਾਗੇ;
    ਜੇ (ggml_is_quantized(dst->type)) {
        ਅਚਲ uint32_t ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = src->ne[0] / ggml_blck_ਮਾਪ(dst->type);
        ਧਾਗੇ =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_ਖੰਡਾਂ ? (ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ / 2) : ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
    } ਨਹੀਂਤਾਂ ਜੇ (decisions->vec4) {
        ਧਾਗੇ = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } ਨਹੀਂਤਾਂ {
        ਧਾਗੇ = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ਧਾਗੇ, decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
ਸਥਿਰ std::vector<wgpu::Constant_ਇੰਦਰਾਜ> ggml_webgpu_wg_ਮਾਪ_ਇੰਦਰਾਜ(uint32_t wg_ਮਾਪ) {
    std::vector<wgpu::Constant_ਇੰਦਰਾਜ> constants(1);
    constants[0].ਕੁੰਜੀ   = "wg_size";
    constants[0].value = wg_ਮਾਪ;
    ਵਾਪਸ constants;
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_get_ਪੰਕਤੀਆਂ(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ਅਚਲ ਤਾਰਕਿਕ float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ਖਾਲੀ_ਸੂਚਕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    ਅਚਲ uint32_t ਵਿੱਥ_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type));
    ਅਚਲ uint32_t ਵਿੱਥ_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type));
    ਅਚਲ uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type));
    ਅਚਲ uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type));
    ਅਚਲ uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type));

    ਅਚਲ ਤਾਰਕਿਕ vec4_aligned = ਵਿੱਥ_src % 4 == 0 && ਵਿੱਥ_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_ਪੰਕਤੀਆਂ_pipeline(shader_lib_ctx, vec4_aligned);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_get_ਪੰਕਤੀਆਂ_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { ਵਿੱਥ_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_ਮਾਪ(idx->type)),
                                     ਵਿੱਥ_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_ਮਾਪ(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_ਮਾਪ(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_ਮਾਪ(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst) };

    uint32_t ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t ਕੁੱਲ_ਧਾਗੇ  = float_parallel ? ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ * ਕੁੱਲ_ਪੰਕਤੀਆਂ : ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ਕੁੱਲ_ਧਾਗੇ, decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ ਖਾਲੀ ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_ਮਾਪ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ਅਚਲ size_t dst_ਵਿੱਥ           = ggml_webgpu_tensor_ਵਿੱਥ(dst);
    ਅਚਲ size_t q8_src1_align_ਵਿੱਥ = ROUNDUP_POW2(
        dst_ਵਿੱਥ + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
    ਅਚਲ size_t q8_src1_binding_ਮਾਪ = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> q8_entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src1),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_ਵਿੱਥ, q8_src1_binding_ਮਾਪ)
    };

    ਸਵੈਚਾਲਿਤ q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_ਮਾਪ     = q8_decisions->wg_ਮਾਪ;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ਅਚਲ uint32_t wg_ਪ੍ਰਤੀ_vec     = (src0->ne[0] / 4 + (q8_wg_ਮਾਪ - 1)) / q8_wg_ਮਾਪ;
    ਅਚਲ uint32_t q8_ਕੁੱਲ_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_ਪ੍ਰਤੀ_vec;
    ਅਚਲ uint32_t max_wg_ਪ੍ਰਤੀ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension;
    compute_2d_workgroups(q8_ਕੁੱਲ_wg, max_wg_ਪ੍ਰਤੀ_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    ਤਾਰਕਿਕ use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    ਤਾਰਕਿਕ use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_ਗੁਣਨਫਲ,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_ਮਾਪ              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_ਮੈਟ੍ਰਿਕਸ = ctx->global_ctx->capabilities.supports_subgroup_ਮੈਟ੍ਰਿਕਸ;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_ਮਾਪ        = ctx->global_ctx->capabilities.min_subgroup_ਮਾਪ;
    shader_lib_ctx.max_subgroup_ਮਾਪ        = ctx->global_ctx->capabilities.max_subgroup_ਮਾਪ;
    shader_lib_ctx.supports_dot_ਗੁਣਨਫਲ     = ctx->global_ctx->capabilities.supports_dot_ਗੁਣਨਫਲ;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    ਅਚਲ ਤਾਰਕਿਕ src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    ਜੇ (use_mat_vec) {
        ਜੇ (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } ਨਹੀਂਤਾਂ {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t ਵਿੱਥ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type));
    uint32_t ਵਿੱਥ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type));
    size_t   merged_ਵਿੱਥ = 0;
    size_t   merged_ਮਾਪ   = 0;
    ਜੇ (src_overlap) {
        ਅਚਲ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_ਵਿੱਥ = merged_range.ਵਿੱਥ;
        merged_ਮਾਪ   = merged_range.size;
        ਵਿੱਥ_src0   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src0, merged_range);
        ਵਿੱਥ_src1   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { ਵਿੱਥ_src0,
                                     ਵਿੱਥ_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {};
    ਜੇ (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0));
        ਸਵੈਚਾਲਿਤ & mmvq_qq8_ਇੰਦਰਾਜ = dispatches[0].bind_ਸਮੂਹ_entries[1];
        entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_ਇੰਦਰਾਜ.ਵਿੱਥ,
                                                            mmvq_qq8_ਇੰਦਰਾਜ.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst));
    } ਨਹੀਂਤਾਂ ਜੇ (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src0), merged_ਵਿੱਥ, merged_ਮਾਪ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst));
    } ਨਹੀਂਤਾਂ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ਅਚਲ uint32_t max_wg_ਪ੍ਰਤੀ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension;

    ਜੇ (use_mat_vec) {
        ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_ਪ੍ਰਤੀ_wg);
        uint32_t ਕੁੱਲ_wg      = output_groups * batches;
        compute_2d_workgroups(ਕੁੱਲ_wg, max_wg_ਪ੍ਰਤੀ_dim, wg_x, wg_y);
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        ਜੇ (decisions->use_subgroup_ਮੈਟ੍ਰਿਕਸ) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_ਮੈਟ੍ਰਿਕਸ_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_ਮੈਟ੍ਰਿਕਸ_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } ਨਹੀਂਤਾਂ {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_ਮਾਪ_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_ਮਾਪ_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t ਕੁੱਲ_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(ਕੁੱਲ_wg, max_wg_ਪ੍ਰਤੀ_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    ਵਾਪਸ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    ਅਚਲ uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ਅਚਲ uint32_t param_n_expert_ਵਰਤਿਆ = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ਮਾਪ(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_ਵਰਤਿਆ,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src0),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src0)),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src1),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src1)),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src2),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src2)),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, dst),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ਅਚਲ uint32_t max_wg_ਪ੍ਰਤੀ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_ਪ੍ਰਤੀ_wg);
    uint32_t       ਕੁੱਲ_wg       = output_groups * param_n_expert_ਵਰਤਿਆ;
    compute_2d_workgroups(ਕੁੱਲ_wg, max_wg_ਪ੍ਰਤੀ_dim, wg_x, wg_y);

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
ਸਥਿਰ uint32_t ggml_webgpu_mul_mat_id_max_tiles(ਅਚਲ ggml_tensor * dst) {
    ਅਚਲ uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    ਅਚਲ uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    ਵਾਪਸ CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_ਮਾਪ_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
ਸਥਿਰ size_t ggml_webgpu_mul_mat_id_tiles_nbytes(ਅਚਲ ggml_tensor * dst) {
    ਵਾਪਸ (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * ਆਕਾਰ(uint32_t);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    ਜੇ (dst->ne[2] == 1) {
        ਵਾਪਸ ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ਅਚਲ uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ਅਚਲ uint32_t param_n_expert_ਵਰਤਿਆ = (uint32_t) dst->ne[1];
    ਅਚਲ uint32_t param_n_ਟੋਕਨ      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ਮਾਪ(src2->type)),
        param_n_expert,
        param_n_expert_ਵਰਤਿਆ,
        param_n_ਟੋਕਨ,
        (uint32_t) (src2->nb[1] / ggml_type_ਮਾਪ(src2->type)),
    };

    ਅਚਲ size_t dst_ਵਿੱਥ       = ggml_webgpu_tensor_ਵਿੱਥ(dst);
    ਅਚਲ size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * ਆਕਾਰ(uint32_t);
    ਅਚਲ size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ਅਚਲ size_t ids_cache_align_ਵਿੱਥ = ROUNDUP_POW2(
        dst_ਵਿੱਥ + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
    ਅਚਲ size_t ids_tiles_align_ਵਿੱਥ =
        ROUNDUP_POW2(ids_cache_align_ਵਿੱਥ + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);

    ਅਚਲ size_t ids_cache_bind_ਮਾਪ = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ਅਚਲ size_t ids_tiles_bind_ਮਾਪ = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> gather_entries = {
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src2),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src2)),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_ਵਿੱਥ, ids_cache_bind_ਮਾਪ),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_ਵਿੱਥ, ids_tiles_bind_ਮਾਪ),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_ਵਰਤਿਆ,
        param_n_ਟੋਕਨ,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> main_entries = {
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src0),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src0)),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src1),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src1)),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, dst),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, dst)),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_ਵਿੱਥ, ids_cache_bind_ਮਾਪ),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_ਵਿੱਥ, ids_tiles_bind_ਮਾਪ),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    ਸਵੈਚਾਲਿਤ * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_ਮਾਪ_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_ਮਾਪ_N);

    ਅਚਲ uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_ਮਾਪ_m;
    ਅਚਲ uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ਅਚਲ uint32_t ਕੁੱਲ_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    ਵਾਪਸ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ਬਣਤਰ ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries;
    size_t                            kv_bind_ਵਿੱਥ = 0;
    size_t                            kv_bind_ਮਾਪ   = 0;
    ਤਾਰਕਿਕ                              has_ਨਕਾਬ       = ਝੂਠ;
    ਤਾਰਕਿਕ                              has_sinks      = ਝੂਠ;
    ਤਾਰਕਿਕ                              kv_overlap     = ਝੂਠ;
};

ਸਥਿਰ ਤਾਰਕਿਕ ggml_webgpu_flash_attn_use_vec_ਰਸਤਾ(ਅਚਲ webgpu_global_context & global_ctx,
                                                ਅਚਲ ggml_tensor *           Q,
                                                ਅਚਲ ggml_tensor *           K,
                                                ਅਚਲ ggml_tensor *           V) {
    ਅਚਲ size_t storage_ਵਿੱਥ_alignment = global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;

    ਅਚਲ ਤਾਰਕਿਕ k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_ਵਿੱਥ_alignment);
    ਅਚਲ ਤਾਰਕਿਕ v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_ਵਿੱਥ_alignment);

    ਅਚਲ uint32_t k_vec_ਸਿਰ_align =
        ggml_is_quantized(K->type) ? ggml_blck_ਮਾਪ(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ;
    ਅਚਲ uint32_t v_vec_ਸਿਰ_align =
        ggml_is_quantized(V->type) ? ggml_blck_ਮਾਪ(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ;
    ਅਚਲ ਤਾਰਕਿਕ kv_vec_ਸਿਰ_dims_aligned = Q->ne[0] % k_vec_ਸਿਰ_align == 0 && V->ne[0] % v_vec_ਸਿਰ_align == 0;

    ਵਾਪਸ global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_ਸਿਰ_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

ਸਥਿਰ ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    ਨਕਾਬ,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    ਦਸ਼ਮਲਵ scale         = ggml_get_op_params_f32(dst, 0);
    ਦਸ਼ਮਲਵ max_bias      = ggml_get_op_params_f32(dst, 1);
    ਦਸ਼ਮਲਵ logit_softcap = ggml_get_op_params_f32(dst, 2);
    ਜੇ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    ਦਸ਼ਮਲਵ n_ਸਿਰ_log2 = ਦਸ਼ਮਲਵ(1u << (uint32_t) floor(log2(Q->ne[2])));
    ਦਸ਼ਮਲਵ m0          = powf(2.0f, -(max_bias) / n_ਸਿਰ_log2);
    ਦਸ਼ਮਲਵ m1          = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = ਨਕਾਬ;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_ਮੈਟ੍ਰਿਕਸ = ctx->global_ctx->capabilities.supports_subgroup_ਮੈਟ੍ਰਿਕਸ;
    op.shader_lib_ctx.max_wg_ਮਾਪ              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
    op.shader_lib_ctx.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_ਮਾਪ;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_ਮਾਪ        = ctx->global_ctx->capabilities.min_subgroup_ਮਾਪ;
    op.shader_lib_ctx.max_subgroup_ਮਾਪ        = ctx->global_ctx->capabilities.max_subgroup_ਮਾਪ;

    op.has_ਨਕਾਬ   = ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ;
    op.has_sinks  = sinks != ਖਾਲੀ_ਸੂਚਕ;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t ਵਿੱਥ_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_ਮਾਪ(K->type));
    uint32_t ਵਿੱਥ_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_ਮਾਪ(V->type));
    ਜੇ (op.kv_overlap) {
        ਅਚਲ ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_ਵਿੱਥ                                   = merged_range.ਵਿੱਥ;
        op.kv_bind_ਮਾਪ                                     = merged_range.size;
        ਵਿੱਥ_k                                            = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(K, merged_range);
        ਵਿੱਥ_v                                            = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_ਮਾਪ(Q->type)),
        ਵਿੱਥ_k,
        ਵਿੱਥ_v,
        op.has_ਨਕਾਬ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ਨਕਾਬ) / ggml_type_ਮਾਪ(ਨਕਾਬ->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_ਮਾਪ(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_ਮਾਪ(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_ਮਾਪ(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_ਮਾਪ(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_ਮਾਪ(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_ਮਾਪ(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_ਮਾਪ(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_ਮਾਪ(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_ਮਾਪ(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_ਮਾਪ(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_ਨਕਾਬ ? (uint32_t) (ਨਕਾਬ->nb[3] / ggml_type_ਮਾਪ(ਨਕਾਬ->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_ਸਿਰ_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, Q),
    };
    ਜੇ (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(K), op.kv_bind_ਵਿੱਥ, op.kv_bind_ਮਾਪ));
    } ਨਹੀਂਤਾਂ {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, V));
    }
    uint32_t binding_ਸੂਚਕਾਂਕ = op.kv_overlap ? 2u : 3u;
    ਜੇ (op.has_ਨਕਾਬ) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਸੂਚਕਾਂਕ++, ਨਕਾਬ));
    }
    ਜੇ (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਸੂਚਕਾਂਕ++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਸੂਚਕਾਂਕ++, dst));

    ਵਾਪਸ op;
}

ਸਥਿਰ uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ਅਚਲ uint64_t kv_ਵਿਸਤਾਰ = (uint64_t) kv_tile;
    ਜਦੋਂ ((2u * nwg * kv_ਵਿਸਤਾਰ) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    ਵਾਪਸ std::min(nwg, vec_nwg_cap);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_flash_attn_ਸਿੱਧਾ(webgpu_context & ctx, ਅਚਲ ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    ਸਵੈਚਾਲਿਤ *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_ਪ੍ਰਤੀ_ਸਿਰ = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_ਪ੍ਰਤੀ_ਸਿਰ * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             ਨਕਾਬ,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::ਬਫਰ_2 blk_buf         = {};
    uint64_t     blk_ਮਾਪ_ਬਾਈਟਾਂ  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_ਗਿਣਤੀ = 0;

    ਅਚਲ uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_ਮਾਪ;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ਅਚਲ uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ਅਚਲ ਤਾਰਕਿਕ     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_ਅਧਾਰ  = 0;
    uint64_t     tmp_ਮਾਪ_ਬਾਈਟਾਂ  = 0;
    wgpu::ਬਫਰ_2 tmp_buf         = {};
    uint64_t     tmp_bind_ਵਿੱਥ = 0;
    uint64_t     tmp_bind_ਮਾਪ   = 0;
    ਅਚਲ size_t align_ਬਾਈਟਾਂ     = ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;
    ਅਚਲ size_t dst_ਵਿੱਥ      = ggml_webgpu_tensor_ਵਿੱਥ(dst);
    size_t       scratch_ਵਿੱਥ  = ROUNDUP_POW2(dst_ਵਿੱਥ + ggml_nbytes(dst), align_ਬਾਈਟਾਂ);

    ਜੇ (use_vec_reduce) {
        ਅਚਲ uint64_t tmp_ਅੰਕੜੇ_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ਅਚਲ uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_ਅਧਾਰ                 = tmp_ਅੰਕੜੇ_elems;
        tmp_ਮਾਪ_ਬਾਈਟਾਂ =
            ROUNDUP_POW2((tmp_ਅੰਕੜੇ_elems + tmp_stats_elems) * ਆਕਾਰ(ਦਸ਼ਮਲਵ), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_ਅਧਾਰ <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_ਵਿੱਥ = scratch_ਵਿੱਥ;
        tmp_bind_ਮਾਪ   = tmp_ਮਾਪ_ਬਾਈਟਾਂ;
        scratch_ਵਿੱਥ  = ROUNDUP_POW2(scratch_ਵਿੱਥ + tmp_ਮਾਪ_ਬਾਈਟਾਂ, align_ਬਾਈਟਾਂ);
    } ਨਹੀਂਤਾਂ {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_ਮਾਪ_ਬਾਈਟਾਂ  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_ਵਿੱਥ = scratch_ਵਿੱਥ;
        tmp_bind_ਮਾਪ   = tmp_ਮਾਪ_ਬਾਈਟਾਂ;
        scratch_ਵਿੱਥ  = ROUNDUP_POW2(scratch_ਵਿੱਥ + tmp_ਮਾਪ_ਬਾਈਟਾਂ, align_ਬਾਈਟਾਂ);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> blk_entries;
    ਜੇ (op.has_ਨਕਾਬ) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ਅਚਲ uint32_t stride_mask3 = (uint32_t) (ਨਕਾਬ->nb[3] / ggml_type_ਮਾਪ(ਨਕਾਬ->type));
        blk_batch_ਗਿਣਤੀ             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ਅਚਲ uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_ਗਿਣਤੀ;
        blk_ਮਾਪ_ਬਾਈਟਾਂ              = ROUNDUP_POW2(blk_elems * ਆਕਾਰ(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        ਅਚਲ ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ਨਕਾਬ) / ggml_type_ਮਾਪ(ਨਕਾਬ->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(ਨਕਾਬ),
                                              ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, ਨਕਾਬ),
                                              ggml_webgpu_tensor_binding_ਮਾਪ(ctx, ਨਕਾਬ)),
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, blk_buf, scratch_ਵਿੱਥ, blk_ਮਾਪ_ਬਾਈਟਾਂ),
        };
        scratch_ਵਿੱਥ = ROUNDUP_POW2(scratch_ਵਿੱਥ + blk_ਮਾਪ_ਬਾਈਟਾਂ, align_ਬਾਈਟਾਂ);
    }

    std::vector<uint32_t> ਵੰਡੋ_params = op.params;
    ਜੇ (op.has_ਨਕਾਬ) {
        ਵੰਡੋ_params.push_back(0u);                     // blk_base
        ਵੰਡੋ_params.push_back(blk_nblk0);              // blk_nblk0
        ਵੰਡੋ_params.push_back(blk_nblk1);              // blk_nblk1
    }
    ਵੰਡੋ_params.push_back(0u);                         // tmp_data_base
    ਵੰਡੋ_params.push_back((uint32_t) tmp_stats_ਅਧਾਰ);  // tmp_stats_base
    ਵੰਡੋ_params.push_back(nwg);                        // nwg

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> ਵੰਡੋ_entries = {
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, Q),
                                          ggml_webgpu_tensor_binding_ਮਾਪ(ctx, Q)),
    };
    ਜੇ (op.kv_overlap) {
        ਵੰਡੋ_entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(K), op.kv_bind_ਵਿੱਥ, op.kv_bind_ਮਾਪ));
    } ਨਹੀਂਤਾਂ {
        ਵੰਡੋ_entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, K),
                                                                  ggml_webgpu_tensor_binding_ਮਾਪ(ctx, K)));
        ਵੰਡੋ_entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, V),
                                                                  ggml_webgpu_tensor_binding_ਮਾਪ(ctx, V)));
    }
    uint32_t ਵੰਡੋ_binding_ਸੂਚਕਾਂਕ = op.kv_overlap ? 2u : 3u;
    ਜੇ (op.has_ਨਕਾਬ) {
        ਵੰਡੋ_entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ਵੰਡੋ_binding_ਸੂਚਕਾਂਕ++, ggml_webgpu_tensor_buf(ਨਕਾਬ),
                                                                  ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, ਨਕਾਬ),
                                                                  ggml_webgpu_tensor_binding_ਮਾਪ(ctx, ਨਕਾਬ)));
    }
    ਜੇ (op.has_sinks) {
        ਵੰਡੋ_entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ਵੰਡੋ_binding_ਸੂਚਕਾਂਕ++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_ਮਾਪ(ctx, sinks)));
    }
    ਜੇ (op.has_ਨਕਾਬ) {
        ਵੰਡੋ_entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ਵੰਡੋ_binding_ਸੂਚਕਾਂਕ++, blk_buf, blk_entries[1].ਵਿੱਥ, blk_ਮਾਪ_ਬਾਈਟਾਂ));
    }
    ਵੰਡੋ_entries.push_back(
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ਵੰਡੋ_binding_ਸੂਚਕਾਂਕ++, tmp_buf, tmp_bind_ਵਿੱਥ, tmp_bind_ਮਾਪ));
    ਵੰਡੋ_entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ਵੰਡੋ_binding_ਸੂਚਕਾਂਕ++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, dst),
                                                              ggml_webgpu_tensor_binding_ਮਾਪ(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> reduce_entries;
    ਜੇ (use_vec_reduce) {
        ਅਚਲ uint32_t reduce_sg_ਮਾਪ = ctx->global_ctx->capabilities.max_subgroup_ਮਾਪ;
        ਅਚਲ uint32_t reduce_wg_ਮਾਪ = std::max(
            reduce_sg_ਮਾਪ,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_ਮਾਪ,
                                          ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_ਮਾਪ                    = reduce_wg_ਮਾਪ;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_ਅਧਾਰ,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, tmp_buf, tmp_bind_ਵਿੱਥ, tmp_ਮਾਪ_ਬਾਈਟਾਂ),
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, dst),
                                              ggml_webgpu_tensor_binding_ਮਾਪ(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ਅਚਲ uint64_t ਵੰਡੋ_wg_ਕੁੱਲ = (uint64_t) wg_x * nwg;
    GGML_ASSERT(ਵੰਡੋ_wg_ਕੁੱਲ <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    ਜੇ (op.has_ਨਕਾਬ) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_ਗਿਣਤੀ }
        });
    }
    dispatches.push_back({
        pipeline, std::move(ਵੰਡੋ_params), std::move(ਵੰਡੋ_entries), { (uint32_t) ਵੰਡੋ_wg_ਕੁੱਲ, 1u }
    });
    ਜੇ (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    ਵਾਪਸ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    ਨਕਾਬ,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, ਨਕਾਬ, sinks, dst);
    ਜੇ (ggml_webgpu_flash_attn_use_vec_ਰਸਤਾ(ctx->global_ctx, Q, K, V)) {
        ਵਾਪਸ ggml_webgpu_flash_attn_vec(ctx, Q, K, V, ਨਕਾਬ, sinks, dst, std::move(op));
    }
    ਵਾਪਸ ggml_webgpu_flash_attn_ਸਿੱਧਾ(ctx, op);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ਤਾਰਕਿਕ is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ਖਾਲੀ_ਸੂਚਕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ਅਚਲ ਤਾਰਕਿਕ inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    ਜੇ (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        ਚੋਣ (unary_op) {
            ਹਾਲਤ GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    ਦਸ਼ਮਲਵ alpha_n = ggml_get_op_params_f32(dst, 1);
                    ਦਸ਼ਮਲਵ alpha_p = ggml_get_op_params_f32(dst, 2);
                    ਦਸ਼ਮਲਵ beta    = ggml_get_op_params_f32(dst, 3);
                    ਦਸ਼ਮਲਵ eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    ਰੋਕੋ;
                }
            ਮੂਲ:
                ਰੋਕੋ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (dst->op == GGML_OP_CLAMP) {
        ਦਸ਼ਮਲਵ clamp_min = ggml_get_op_params_f32(dst, 0);
        ਦਸ਼ਮਲਵ clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } ਨਹੀਂਤਾਂ ਜੇ (dst->op == GGML_OP_FILL) {
        ਦਸ਼ਮਲਵ fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, effective_src),
    };
    ਜੇ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_ਦੁਆਧਾਰੀ_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    ਅਚਲ ਤਾਰਕਿਕ      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_ਦੁਆਧਾਰੀ_pipeline(shader_lib_ctx, src_overlap);
    ਸਵੈਚਾਲਿਤ *          decisions   = static_cast<ggml_webgpu_ਦੁਆਧਾਰੀ_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_ਵਿੱਥ = ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src0);
    size_t src1_webgpu_tensor_align_ਵਿੱਥ = ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src1);

    uint32_t ਵਿੱਥ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type));
    uint32_t ਵਿੱਥ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type));
    size_t   merged_ਵਿੱਥ = 0;
    size_t   merged_ਮਾਪ   = 0;
    ਜੇ (decisions->src_overlap) {
        ਅਚਲ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_ਵਿੱਥ = merged_range.ਵਿੱਥ;
        merged_ਮਾਪ   = merged_range.size;
        ਵਿੱਥ_src0   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src0, merged_range);
        ਵਿੱਥ_src1   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        ਵਿੱਥ_src0,
        ਵਿੱਥ_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries;

    ਜੇ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src0), merged_ਵਿੱਥ, merged_ਮਾਪ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst));
    } ਨਹੀਂਤਾਂ {
        entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_ਵਿੱਥ,
                                                            ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_ਵਿੱਥ,
                                                            ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src1)));
        ਜੇ (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_ਜੋੜੋ_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ਜੋੜੋ_id_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ਮਾਪ(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_ਮਾਪ(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_ਮਾਪ(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, src2));

    ਜੇ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       ਕੁੱਲ_wg       = ggml_nrows(dst);
    ਅਚਲ uint32_t max_wg_ਪ੍ਰਤੀ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension;
    compute_2d_workgroups(ਕੁੱਲ_wg, max_wg_ਪ੍ਰਤੀ_dim, wg_x, wg_y);

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    ਜੇ (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        ਵਾਪਸ {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    ਅਚਲ ਤਾਰਕਿਕ      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    ਸਵੈਚਾਲਿਤ *          decisions   = static_cast<ggml_webgpu_ਦੁਆਧਾਰੀ_shader_decisions *>(pipeline.context.get());

    uint32_t ਵਿੱਥ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type));
    uint32_t ਵਿੱਥ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type));
    size_t   merged_ਵਿੱਥ = 0;
    size_t   merged_ਮਾਪ   = 0;
    ਜੇ (decisions->src_overlap) {
        ਜੇ (ggml_nbytes(src0) == 0) {
            merged_ਵਿੱਥ = ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src1);
            merged_ਮਾਪ   = ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src1);
            ਵਿੱਥ_src0   = 0;
            ਵਿੱਥ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type));
        } ਨਹੀਂਤਾਂ ਜੇ (ggml_nbytes(src1) == 0) {
            merged_ਵਿੱਥ = ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src0);
            merged_ਮਾਪ   = ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src0);
            ਵਿੱਥ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type));
            ਵਿੱਥ_src1   = 0;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_ਵਿੱਥ = merged_range.ਵਿੱਥ;
            merged_ਮਾਪ   = merged_range.size;
            ਵਿੱਥ_src0   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src0, merged_range);
            ਵਿੱਥ_src1   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     ਵਿੱਥ_src0,
                                     ਵਿੱਥ_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_ਮਾਪ(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {};
    ਜੇ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src0), merged_ਵਿੱਥ, merged_ਮਾਪ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst));
    } ਨਹੀਂਤਾਂ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ne, decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ne, decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    ਜੇ (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t ਵਿੱਥ_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_ਮਾਪ(rn_src->type));
    uint32_t ਵਿੱਥ_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_ਮਾਪ(mul_src->type));
    size_t merged_ਵਿੱਥ = 0;
    size_t merged_ਮਾਪ   = 0;

    std::vector<uint32_t> params = {
        ਵਿੱਥ_rn_src,
        ਵਿੱਥ_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_ਮਾਪ(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_ਮਾਪ(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_ਮਾਪ(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_ਮਾਪ(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_ਮਾਪ(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_ਮਾਪ(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) mul_src->ne[0],
        (uint32_t) mul_src->ne[1],
        (uint32_t) mul_src->ne[2],
        (uint32_t) mul_src->ne[3],
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) dst->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(rn_dst, 0))  // epsilon, treated as f32 in the shader
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    ਅਚਲ ਤਾਰਕਿਕ      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    ਸਵੈਚਾਲਿਤ *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    ਜੇ (decisions->src_overlap) {
        ਅਚਲ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_ਵਿੱਥ  = merged_range.ਵਿੱਥ;
        merged_ਮਾਪ    = merged_range.size;
        ਵਿੱਥ_rn_src  = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(rn_src, merged_range);
        ਵਿੱਥ_mul_src = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(mul_src, merged_range);
        params[0]      = ਵਿੱਥ_rn_src;
        params[1]      = ਵਿੱਥ_mul_src;
    }

    ਜੇ (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, mul_src));
    } ਨਹੀਂਤਾਂ ਜੇ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(rn_src), merged_ਵਿੱਥ, merged_ਮਾਪ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst));
    } ਨਹੀਂਤਾਂ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x,
                          wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_ਪੰਕਤੀ_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ਪੰਕਤੀ_norm_pipeline(shader_lib_ctx);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src) };
    ਜੇ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x,
                          wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ਅਚਲ ਤਾਰਕਿਕ inplace         = decisions->inplace;
    ਅਚਲ ਪੂਰਨ  has_freq_ਗੁਣਕ = (src2 != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ ਪੂਰਨ n_dims     = ((int32_t *) dst->op_params)[1];
    ਅਚਲ ਪੂਰਨ mode       = ((int32_t *) dst->op_params)[2];
    ਅਚਲ ਪੂਰਨ n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ਅਚਲ ਪੂਰਨ n_offs     = ((int32_t *) dst->op_params)[15];

    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ freq_scale;
    ਦਸ਼ਮਲਵ ext_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ attn_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ beta_fast;
    ਦਸ਼ਮਲਵ beta_slow;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ, (int32_t *) dst->op_params + 5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale, (int32_t *) dst->op_params + 6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ, (int32_t *) dst->op_params + 7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, (int32_t *) dst->op_params + 8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast, (int32_t *) dst->op_params + 9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow, (int32_t *) dst->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਪੂਰਨ sections[4];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(sections, (int32_t *) dst->op_params + 11, 4 * ਆਕਾਰ(ਪੂਰਨ));

    ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    ਦਸ਼ਮਲਵ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)),
        src2 != ਖਾਲੀ_ਸੂਚਕ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ਮਾਪ(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_ਗੁਣਕ),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_ਗੁਣਕ),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries     = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    ਜੇ (has_freq_ਗੁਣਕ) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 2, src2));
    }
    ਜੇ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    ਅਚਲ ਤਾਰਕਿਕ      src_overlap = src1 != ਖਾਲੀ_ਸੂਚਕ && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    ਸਵੈਚਾਲਿਤ * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ਅਚਲ ਪੂਰਨ ਵੰਡੋ = (src1 != ਖਾਲੀ_ਸੂਚਕ);

    uint32_t ਵਿੱਥ_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type));
    uint32_t ਵਿੱਥ_src1 =
        src1 != ਖਾਲੀ_ਸੂਚਕ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)) : 0;
    size_t merged_ਵਿੱਥ = 0;
    size_t merged_ਮਾਪ   = 0;
    ਜੇ (decisions->src_overlap) {
        ਅਚਲ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_ਵਿੱਥ = merged_range.ਵਿੱਥ;
        merged_ਮਾਪ   = merged_range.size;
        ਵਿੱਥ_src0   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src0, merged_range);
        ਵਿੱਥ_src1   = ggml_webgpu_tensor_merged_element_ਵਿੱਥ(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ਵਿੱਥ_src0,
        ਵਿੱਥ_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
        src1 != ਖਾਲੀ_ਸੂਚਕ ? (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        src1 != ਖਾਲੀ_ਸੂਚਕ ? (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        src1 != ਖਾਲੀ_ਸੂਚਕ ? (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  //ਹੱਦਬੰਦੀ
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries;
    uint32_t                          dst_binding = 1;
    ਜੇ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(0, ggml_webgpu_tensor_buf(src0), merged_ਵਿੱਥ, merged_ਮਾਪ));
    } ਨਹੀਂਤਾਂ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src0));
    }
    ਜੇ (ਵੰਡੋ && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ਖਾਲੀ_ਸੂਚਕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src) };

    ਜੇ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ਅਚਲ ਤਾਰਕਿਕ inplace     = decisions->inplace;
    ਅਚਲ ਪੂਰਨ  has_ਨਕਾਬ    = (src1 != ਖਾਲੀ_ਸੂਚਕ);
    ਅਚਲ ਪੂਰਨ  has_sink    = (src2 != ਖਾਲੀ_ਸੂਚਕ);
    ਦਸ਼ਮਲਵ      max_bias    = ggml_get_op_params_f32(dst, 1);
    ਦਸ਼ਮਲਵ      n_ਸਿਰ_log2 = ਦਸ਼ਮਲਵ(1u << (uint32_t) floor(log2(src0->ne[2])));
    ਦਸ਼ਮਲਵ      m0          = powf(2.0f, -(max_bias) / n_ਸਿਰ_log2);
    ਦਸ਼ਮਲਵ      m1          = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ਮਾਪ(src0->type)),
        has_ਨਕਾਬ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ਮਾਪ(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ਮਾਪ(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ਮਾਪ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ਮਾਪ(src0->type)),
        has_ਨਕਾਬ ? (uint32_t) (src1->nb[1] / ggml_type_ਮਾਪ(src1->type)) : 0,
        has_ਨਕਾਬ ? (uint32_t) (src1->nb[2] / ggml_type_ਮਾਪ(src1->type)) : 0,
        has_ਨਕਾਬ ? (uint32_t) (src1->nb[3] / ggml_type_ਮਾਪ(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_ਨਕਾਬ ? (uint32_t) src1->ne[2] : 0,
        has_ਨਕਾਬ ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_ਸਿਰ_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries     = { ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src0),
        ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src0)) };
    uint32_t                          binding_ਅੰਕ = 1;
    ਜੇ (has_ਨਕਾਬ) {
        entries.push_back(ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(binding_ਅੰਕ, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, src1),
                                                            ggml_webgpu_tensor_binding_ਮਾਪ(ctx, src1)));
        binding_ਅੰਕ++;
    }
    ਜੇ (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ, src2));
        binding_ਅੰਕ++;
    }
    ਜੇ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, binding_ਅੰਕ, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = ggml_nrows(dst);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ਤਾਰਕਿਕ is_ਸਿਖਰ_k = dst->op == GGML_OP_ਸਿਖਰ_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ਖਾਲੀ_ਸੂਚਕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
    shader_lib_ctx.wg_mem_ਹੱਦਬੰਦੀ_ਬਾਈਟਾਂ = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_ਮਾਪ;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    ਸਵੈਚਾਲਿਤ * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_ਮਿਲਾਓ_pipeline = ctx->shader_lib->get_argsort_ਮਿਲਾਓ_pipeline(shader_lib_ctx);

    ਅਚਲ uint32_t src_ne0 = (uint32_t) src->ne[0];
    ਅਚਲ uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ਅਚਲ uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_ਮਾਪ);
    ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 =
        is_ਸਿਖਰ_k ? std::min(argsort_decisions->wg_ਮਾਪ, (uint32_t) dst->ne[0]) : argsort_decisions->wg_ਮਾਪ;
    uint32_t ਬਾਹਰਲਾ_ne0 = src_ne0;
    ਜੇ (is_ਸਿਖਰ_k) {
        ਜੇ (npr > 1) {
            ਅਚਲ uint32_t ਆਖਰੀ_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_ਮਾਪ;
            ਬਾਹਰਲਾ_ne0                  = (npr - 1) * ਖੰਡ_ਮਾਪ_3 + std::min(ਆਖਰੀ_tile, ਖੰਡ_ਮਾਪ_3);
        } ਨਹੀਂਤਾਂ {
            ਬਾਹਰਲਾ_ne0 = ਖੰਡ_ਮਾਪ_3;
        }
    }

    uint32_t ਮਿਲਾਓ_len    = ਖੰਡ_ਮਾਪ_3;
    uint32_t ਮਿਲਾਓ_passes = 0;
    ਜਦੋਂ (ਮਿਲਾਓ_len < ਬਾਹਰਲਾ_ne0) {
        ਮਿਲਾਓ_len <<= 1;
        ਮਿਲਾਓ_passes++;
    }

    ਅਚਲ ਤਾਰਕਿਕ ਸ਼ੁਰੂ_ਵਿੱਚ_tmp = (ਮਿਲਾਓ_passes % 2) == 1;

    ਅਚਲ size_t dst_ਵਿੱਥ = ggml_webgpu_tensor_ਵਿੱਥ(dst);
    ਅਚਲ size_t idx_nbytes = ਬਾਹਰਲਾ_ne0 * ggml_nrows(dst) * ਆਕਾਰ(int32_t);
    ਅਚਲ size_t tmp_ਵਿੱਥ =
        ROUNDUP_POW2(dst_ਵਿੱਥ + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment);
    ਅਚਲ size_t tmp_binding_ਮਾਪ = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ਅਚਲ size_t dst_binding_ਮਾਪ =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ਅਚਲ uint32_t ਵਿੱਥ_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type));
    ਅਚਲ uint32_t ਵਿੱਥ_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type));
    ਅਚਲ uint32_t ਵਿੱਥ_tmp  = 0;
    ਅਚਲ uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type));
    ਅਚਲ uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type));
    ਅਚਲ uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type));
    ਅਚਲ uint32_t stride_idx1 = ਬਾਹਰਲਾ_ne0;
    ਅਚਲ uint32_t stride_idx2 = ਬਾਹਰਲਾ_ne0 * (uint32_t) dst->ne[1];
    ਅਚਲ uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ਅਚਲ uint32_t init_ਵਿੱਥ       = ਸ਼ੁਰੂ_ਵਿੱਚ_tmp ? ਵਿੱਥ_tmp : ਵਿੱਥ_dst;
    ਅਚਲ size_t   init_align_ਵਿੱਥ = ਸ਼ੁਰੂ_ਵਿੱਚ_tmp ? tmp_ਵਿੱਥ : ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, dst);
    ਅਚਲ size_t   init_binding_ਮਾਪ = ਸ਼ੁਰੂ_ਵਿੱਚ_tmp ? tmp_binding_ਮਾਪ : dst_binding_ਮਾਪ;

    std::vector<uint32_t> init_params = {
        ਵਿੱਥ_src,  init_ਵਿੱਥ, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], ਬਾਹਰਲਾ_ne0,
        ਖੰਡ_ਮਾਪ_3,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ਅਚਲ uint32_t ਕੁੱਲ_wg_init  = npr * nrows;
    ਅਚਲ uint32_t max_wg_ਪ੍ਰਤੀ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension;
    compute_2d_workgroups(ਕੁੱਲ_wg_init, max_wg_ਪ੍ਰਤੀ_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> init_entries = {
        ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
        ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(dst), init_align_ਵਿੱਥ, init_binding_ਮਾਪ)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    ਜੇ (ਮਿਲਾਓ_passes == 0) {
        ਵਾਪਸ ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    ਤਾਰਕਿਕ     ਵਿੱਚ_is_tmp = ਸ਼ੁਰੂ_ਵਿੱਚ_tmp;
    uint32_t len       = ਖੰਡ_ਮਾਪ_3;
    ਜਦੋਂ (len < ਬਾਹਰਲਾ_ne0) {
        ਅਚਲ uint32_t nm = CEIL_DIV(ਬਾਹਰਲਾ_ne0, 2 * len);

        ਅਚਲ ਤਾਰਕਿਕ     ਬਾਹਰਲਾ_is_tmp  = !ਵਿੱਚ_is_tmp;
        ਅਚਲ uint32_t ਵਿੱਥ_ਵਿੱਚ   = ਵਿੱਚ_is_tmp ? ਵਿੱਥ_tmp : ਵਿੱਥ_dst;
        ਅਚਲ uint32_t ਵਿੱਥ_ਬਾਹਰਲਾ  = ਬਾਹਰਲਾ_is_tmp ? ਵਿੱਥ_tmp : ਵਿੱਥ_dst;
        ਅਚਲ size_t   align_ਵਿੱਚ    = ਵਿੱਚ_is_tmp ? tmp_ਵਿੱਥ : ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, dst);
        ਅਚਲ size_t   align_ਬਾਹਰਲਾ   = ਬਾਹਰਲਾ_is_tmp ? tmp_ਵਿੱਥ : ggml_webgpu_tensor_align_ਵਿੱਥ(ctx, dst);
        ਅਚਲ size_t   ਮਾਪ_ਵਿੱਚ     = ਵਿੱਚ_is_tmp ? tmp_binding_ਮਾਪ : dst_binding_ਮਾਪ;
        ਅਚਲ size_t   ਮਾਪ_ਬਾਹਰਲਾ    = ਬਾਹਰਲਾ_is_tmp ? tmp_binding_ਮਾਪ : dst_binding_ਮਾਪ;
        ਅਚਲ uint32_t ਸਿਖਰ_k_ਬਾਹਰਲਾ   = (is_ਸਿਖਰ_k && nm == 1) ? (uint32_t) dst->ne[0] : ਬਾਹਰਲਾ_ne0;
        ਅਚਲ uint32_t stride_out1 = ਸਿਖਰ_k_ਬਾਹਰਲਾ;
        ਅਚਲ uint32_t stride_out2 = ਸਿਖਰ_k_ਬਾਹਰਲਾ * (uint32_t) dst->ne[1];
        ਅਚਲ uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> ਮਿਲਾਓ_params = { ਵਿੱਥ_src,
                                               ਵਿੱਥ_ਵਿੱਚ,
                                               ਵਿੱਥ_ਬਾਹਰਲਾ,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               ਬਾਹਰਲਾ_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               ਸਿਖਰ_k_ਬਾਹਰਲਾ,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> ਮਿਲਾਓ_entries = {
            ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(1, ggml_webgpu_tensor_buf(dst), align_ਵਿੱਚ, ਮਾਪ_ਵਿੱਚ),
            ggml_webgpu_make_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(2, ggml_webgpu_tensor_buf(dst), align_ਬਾਹਰਲਾ, ਮਾਪ_ਬਾਹਰਲਾ)
        };

        uint32_t       wg_x_ਮਿਲਾਓ;
        uint32_t       wg_y_ਮਿਲਾਓ;
        ਅਚਲ uint32_t ਕੁੱਲ_wg_ਮਿਲਾਓ = nm * nrows;
        compute_2d_workgroups(ਕੁੱਲ_wg_ਮਿਲਾਓ, max_wg_ਪ੍ਰਤੀ_dim, wg_x_ਮਿਲਾਓ, wg_y_ਮਿਲਾਓ);

        dispatches.push_back({
            argsort_ਮਿਲਾਓ_pipeline, std::move(ਮਿਲਾਓ_params), std::move(ਮਿਲਾਓ_entries), { wg_x_ਮਿਲਾਓ, wg_y_ਮਿਲਾਓ }
        });

        len <<= 1;
        ਵਿੱਚ_is_tmp = !ਵਿੱਚ_is_tmp;
    }

    ਵਾਪਸ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ਖਾਲੀ_ਸੂਚਕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        ਕੁੱਲ_wg = ggml_nrows(dst);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_ਜੋੜ_ਪੰਕਤੀਆਂ(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ਤਾਰਕਿਕ                  ਕੁੱਲ_ਜੋੜ = dst->op == GGML_OP_ਜੋੜ;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),
                                     ਕੁੱਲ_ਜੋੜ ? 0 : (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)),
                                     ਕੁੱਲ_ਜੋੜ ? 0 : (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)),
                                     ਕੁੱਲ_ਜੋੜ ? 0 : (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)),
                                     ਕੁੱਲ_ਜੋੜ ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     ਕੁੱਲ_ਜੋੜ ? 1 : (uint32_t) src->ne[1],
                                     ਕੁੱਲ_ਜੋੜ ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ਜੋੜ_ਪੰਕਤੀਆਂ_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = ਕੁੱਲ_ਜੋੜ ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);
    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_webgpu_can_fuse_rms_norm_mul(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        ਵਾਪਸ ਝੂਠ;
    }

    // additional constraints specific to this fusion
    ਅਚਲ ggml_tensor * rms_norm = cgraph->nodes[ਗੰਢ_idx];
    ਅਚਲ ggml_tensor * mul      = cgraph->nodes[ਗੰਢ_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    ਜੇ (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    ਜੇ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        ਵਾਪਸ ਝੂਠ;
    }
    // rms_norm shader assumes contiguous rows
    ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(mul->src[0]) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(mul->src[1])) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_webgpu_can_fuse_gdn_cache(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਪੂਰਨ & ਅੰਕ_encoded_ops) {
    ਅਚਲ ggml_tensor * gdn = cgraph->nodes[ਗੰਢ_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ਜੇ (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_tensor * src_v     = gdn->src[2];
    ਅਚਲ int64_t       S_v       = src_v->ne[0];
    ਅਚਲ int64_t       H         = src_v->ne[1];
    ਅਚਲ int64_t       n_ਟੋਕਨ  = src_v->ne[2];
    ਅਚਲ int64_t       n_seqs    = src_v->ne[3];
    ਅਚਲ int64_t       D         = S_v * S_v * H;
    ਅਚਲ int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    ਅਚਲ int64_t       n_written = std::min<int64_t>(n_ਟੋਕਨ, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ਅਚਲ size_t ਪੂਛ_off = ggml_ਪੰਕਤੀ_ਮਾਪ(GGML_TYPE_F32, S_v * H * n_ਟੋਕਨ * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ਅਚਲ ggml_tensor * cpy     = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ                 cpy_idx = 0;
    ਲਈ (ਪੂਰਨ j = ਗੰਢ_idx + 1; j < cgraph->n_nodes && cpy == ਖਾਲੀ_ਸੂਚਕ; ++j) {
        ਅਚਲ ggml_tensor * n = cgraph->nodes[j];
        ਜੇ (ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(n->op) || ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(n)) {
            ਜਾਰੀ;
        }
        ਜੇ (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            ਵਾਪਸ ਝੂਠ;
        }
        cpy     = n;
        cpy_idx = j;
    }
    ਜੇ (cpy == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ਅਚਲ ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ਜੇ (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != ਪੂਛ_off ||
        !ggml_is_contiguous(cpy_src)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ਅਚਲ std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    ਜੇ (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_ਮਾਪ(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_ਪੰਕਤੀ_ਮਾਪ(GGML_TYPE_F32, D)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅੰਕ_encoded_ops = cpy_idx - ਗੰਢ_idx + 1;

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ਅਚਲ uint32_t        ਬਹੁਲਕ_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ਮਾਪ(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_ਮਾਪ(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_ਮਾਪ(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_ਮਾਪ(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_ਮਾਪ(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     ਬਹੁਲਕ_flags };

    std::vector<wgpu::Bind_ਸਮੂਹ_ਇੰਦਰਾਜ> entries = { ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ਸਮੂਹ_ਇੰਦਰਾਜ(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ਮਾਪ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    ਸਵੈਚਾਲਿਤ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ਕੁੱਲ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ਮਾਪ);
    compute_2d_workgroups(ਕੁੱਲ_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension, wg_x, wg_y);

    ਵਾਪਸ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
ਸਥਿਰ std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           ਪੂਰਨ            ਗੰਢ_idx,
                                                           ਪੂਰਨ &          ਅੰਕ_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  ਗੰਢ  = nodes[ਗੰਢ_idx];

    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ)) {
        ਵਾਪਸ std::nullopt;
    }
    ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ਵਾਪਸ std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << ਗੰਢ << ", " << ggml_op_name(ਗੰਢ->op) << ")");

    ggml_tensor * src0 = ਗੰਢ->src[0];
    ggml_tensor * src1 = ਗੰਢ->src[1];
    ggml_tensor * src2 = ਗੰਢ->src[2];

    ਚੋਣ (ਗੰਢ->op) {
            // no-ops
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_RESHAPE:
            ਵਾਪਸ std::nullopt;
        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_CPY:
        ਹਾਲਤ GGML_OP_CONT:
            ਵਾਪਸ ggml_webgpu_cpy(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            ਵਾਪਸ ggml_webgpu_ਸੈੱਟ(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ ggml_webgpu_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ ggml_webgpu_get_ਪੰਕਤੀਆਂ(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ਵਾਪਸ ggml_webgpu_mul_mat(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            ਵਾਪਸ ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, ਗੰਢ);
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ਵਾਪਸ ggml_webgpu_flash_attn(ctx, src0, src1, src2, ਗੰਢ->src[3], ਗੰਢ->src[4], ਗੰਢ);
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
            ਵਾਪਸ ggml_webgpu_ਦੁਆਧਾਰੀ_op(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
            ਵਾਪਸ ggml_webgpu_ਜੋੜੋ_id(ctx, src0, src1, src2, ਗੰਢ);
        ਹਾਲਤ GGML_OP_CONCAT:
            ਵਾਪਸ ggml_webgpu_concat(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_REPEAT:
            ਵਾਪਸ ggml_webgpu_repeat(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_RMS_NORM:
            ਜੇ (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, ਗੰਢ_idx)) {
                ਅੰਕ_encoded_ops        = 2;
                ggml_tensor * mul_ਗੰਢ = nodes[ਗੰਢ_idx + 1];
                ਵਾਪਸ ggml_webgpu_rms_norm_mul(ctx, src0, ਗੰਢ, mul_ਗੰਢ->src[0], mul_ਗੰਢ->src[1], mul_ਗੰਢ);
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ ggml_webgpu_ਪੰਕਤੀ_norm(ctx, src0, ਗੰਢ);
            }
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_L2_NORM:
            ਵਾਪਸ ggml_webgpu_ਪੰਕਤੀ_norm(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_ROPE:
            ਵਾਪਸ ggml_webgpu_rope(ctx, src0, src1, src2, ਗੰਢ);
        ਹਾਲਤ GGML_OP_GLU:
            ਵਾਪਸ ggml_webgpu_glu(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_SCALE:
            ਵਾਪਸ ggml_webgpu_scale(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            ਵਾਪਸ ggml_webgpu_soft_max(ctx, src0, src1, src2, ਗੰਢ);
        ਹਾਲਤ GGML_OP_UNARY:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_FILL:
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_SIN:
        ਹਾਲਤ GGML_OP_COS:
        ਹਾਲਤ GGML_OP_DIAG:
        ਹਾਲਤ GGML_OP_TRI:
            ਵਾਪਸ ggml_webgpu_unary_op(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            ਵਾਪਸ ggml_webgpu_solve_tri(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_SSM_CONV:
            ਵਾਪਸ ggml_webgpu_ssm_conv(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            ਵਾਪਸ ggml_webgpu_ssm_ਜਾਂਚ(ctx, src0, src1, src2, ਗੰਢ->src[3], ਗੰਢ->src[4], ਗੰਢ->src[5], ਗੰਢ->src[6],
                                        ਗੰਢ);
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            ਜੇ (ggml_webgpu_can_fuse_gdn_cache(cgraph, ਗੰਢ_idx, ਅੰਕ_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[ਗੰਢ_idx + ਅੰਕ_encoded_ops - 1]->src[1];
                ਵਾਪਸ ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ਗੰਢ->src[3], ਗੰਢ->src[4], ਗੰਢ->src[5],
                                                   ਗੰਢ, dst_fuse);
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ਗੰਢ->src[3], ਗੰਢ->src[4], ਗੰਢ->src[5],
                                                   ਗੰਢ, ਖਾਲੀ_ਸੂਚਕ);
            }
        ਹਾਲਤ GGML_OP_PAD:
            ਵਾਪਸ ggml_webgpu_pad(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_ARGMAX:
            ਵਾਪਸ ggml_webgpu_argmax(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_ARGSORT:
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            // we reuse the same argsort implementation for top_k
            ਵਾਪਸ ggml_webgpu_argsort(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_CUMSUM:
            ਵਾਪਸ ggml_webgpu_cumsum(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_ਜੋੜ:
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ ggml_webgpu_ਜੋੜ_ਪੰਕਤੀਆਂ(ctx, src0, ਗੰਢ);
        ਹਾਲਤ GGML_OP_CONV_2D:
            ਵਾਪਸ ggml_webgpu_conv_2d(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
            ਵਾਪਸ ggml_webgpu_conv_2d_dw(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            ਵਾਪਸ ggml_webgpu_im2col(ctx, src0, src1, ਗੰਢ);
        ਹਾਲਤ GGML_OP_UPSCALE:
            ਵਾਪਸ ggml_webgpu_upscale(ctx, src0, ਗੰਢ);
        ਮੂਲ:
            ਵਾਪਸ std::nullopt;
    }
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ਅਚਲ std::vector<std::string> & pipeline_ਨਾਂ,
                                                        uint32_t &                       ਅੰਕ_inflight_batches) {
    ਜੇ (pipeline_ਨਾਂ.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Resolve_Query_ਸੈੱਟ(ctx->profile_timestamp_query_ਸੈੱਟ, 0, ctx->profile_timestamp_query_ਗਿਣਤੀ,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_ਬਫਰ_ਮੰਜ਼ਿਲ_ਬਫਰ(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_ਗਿਣਤੀ * ਆਕਾਰ(uint64_t));

    wgpu::Command_ਬਫਰ profile_commands = encoder.ਅੰਤ_2();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, ਅੰਕ_inflight_batches);

    ਅਚਲ size_t mapped_ਮਾਪ = ctx->profile_timestamp_query_ਗਿਣਤੀ * ਆਕਾਰ(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_ਗਿਣਤੀ == 2 * pipeline_ਨਾਂ.size());

    ggml_backend_webgpu_map_ਬਫਰ(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::Map_ਬਹੁਲਕ::Read, 0,
                                   mapped_ਮਾਪ);
    ਅਚਲ uint64_t * ts_ਅੰਕੜੇ = (ਅਚਲ uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_ਮਾਪ);

    ਲਈ (size_t i = 0; i < pipeline_ਨਾਂ.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ਅਚਲ ਦੁੱਗਣਾ elapsed_ms = ਦੁੱਗਣਾ(ts_ਅੰਕੜੇ[2 * i + 1] - ts_ਅੰਕੜੇ[2 * i]) * 1e-6;
        ctx->shader_gpu_ਸਮਾਂ_ms[pipeline_ਨਾਂ[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#ਜੇ_ਅੰਤ

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_check_ਸੈੱਟ_ਪੰਕਤੀਆਂ(webgpu_context & ctx, uint32_t & ਅੰਕ_inflight_batches) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_CHECK_ਸੈੱਟ_ਪੰਕਤੀਆਂ
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_ਬਫਰ_ਮੰਜ਼ਿਲ_ਬਫਰ(ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf, 0, ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf, 0,
                               ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf.Get_ਮਾਪ());
    wgpu::Command_ਬਫਰ commands = encoder.ਅੰਤ_2();
    ggml_backend_webgpu_submit_commands(ctx, commands, ਅੰਕ_inflight_batches);
    ggml_backend_webgpu_map_ਬਫਰ(ctx->global_ctx, ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf, wgpu::Map_ਬਹੁਲਕ::Read, 0,
                                   ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf.Get_ਮਾਪ());
    ਅਚਲ uint32_t * error_ਅੰਕੜੇ = (ਅਚਲ uint32_t *) ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf.GetConstMappedRange();
    ਜੇ (*error_ਅੰਕੜੇ) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf.Unmap();
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(ਅੰਕ_inflight_batches);
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_webgpu_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(ਗ੍ਰਾਫ_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t ਅੰਕ_batched_kernels  = 0;
    uint32_t ਅੰਕ_inflight_batches = 0;
    ਤਾਰਕਿਕ     ਸ਼ਾਮਲ_ਹੈ_ਸੈੱਟ_ਪੰਕਤੀਆਂ    = ਝੂਠ;
    ਪੂਰਨ      ਅੰਕ_encoded_ops      = 1;
    ਪੂਰਨ      ਗੰਢ_idx             = 0;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_ਗਿਣਤੀ = 0;
    std::vector<std::string> profile_pipeline_ਨਾਂ;
#ਜੇ_ਅੰਤ

    ctx->ਸਰਗਰਮ_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    ਜੇ (ctx->batch_compute_passes) {
        ctx->ਸਰਗਰਮ_compute_pass = ctx->ਸਰਗਰਮ_command_encoder.BeginComputePass();
    }

    ਜਦੋਂ (ਗੰਢ_idx < cgraph->n_nodes) {
        ਜੇ (cgraph->nodes[ਗੰਢ_idx]->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
            ਸ਼ਾਮਲ_ਹੈ_ਸੈੱਟ_ਪੰਕਤੀਆਂ = ਸੱਚ;
        }
        ਜੇ (ਸਵੈਚਾਲਿਤ cmd = ggml_webgpu_encode(ctx, cgraph, ਗੰਢ_idx, ਅੰਕ_encoded_ops)) {
            commands.push_back(*cmd);
            ਅੰਕ_batched_kernels += cmd.value().ਅੰਕ_kernels;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_ਨਾਂ.insert(profile_pipeline_ਨਾਂ.end(), cmd->pipeline_ਨਾਂ.begin(),
                                          cmd->pipeline_ਨਾਂ.end());
#ਜੇ_ਅੰਤ
        }

        ਜੇ (ਅੰਕ_batched_kernels >= ctx->global_ctx->command_submit_batch_ਮਾਪ) {
            ਜੇ (ctx->ਸਰਗਰਮ_compute_pass) {
                ctx->ਸਰਗਰਮ_compute_pass.ਸਿਰਾ_2();
            }
            ਅੰਕ_batched_kernels                = 0;
            wgpu::Command_ਬਫਰ batch_commands = ctx->ਸਰਗਰਮ_command_encoder.ਅੰਤ_2();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, ਅੰਕ_inflight_batches);

            // reset state for next batch
            ctx->ਸਰਗਰਮ_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            ਜੇ (ctx->batch_compute_passes) {
                ctx->ਸਰਗਰਮ_compute_pass = ctx->ਸਰਗਰਮ_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            ਜੇ (ctx->profile_timestamp_query_ਗਿਣਤੀ + 2 * ctx->global_ctx->command_submit_batch_ਮਾਪ >=
                WEBGPU_MAX_PROFILE_QUERY_ਗਿਣਤੀ) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_ਨਾਂ, ਅੰਕ_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_ਗਿਣਤੀ = 0;
                profile_pipeline_ਨਾਂ.clear();
            }
#ਜੇ_ਅੰਤ
        }

        ਗੰਢ_idx += ਅੰਕ_encoded_ops;
        ਅੰਕ_encoded_ops = 1;
    }

    ਜੇ (ctx->ਸਰਗਰਮ_compute_pass) {
        ctx->ਸਰਗਰਮ_compute_pass.ਸਿਰਾ_2();
        ctx->ਸਰਗਰਮ_compute_pass = ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (ਅੰਕ_batched_kernels > 0) {
        wgpu::Command_ਬਫਰ batch_commands = ctx->ਸਰਗਰਮ_command_encoder.ਅੰਤ_2();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, ਅੰਕ_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->ਸਰਗਰਮ_command_encoder = ਖਾਲੀ_ਸੂਚਕ;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_ਨਾਂ, ਅੰਕ_inflight_batches);
#ਜੇ_ਅੰਤ

    ਜੇ (ਸ਼ਾਮਲ_ਹੈ_ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
        ggml_backend_webgpu_check_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, ਅੰਕ_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(ਗ੍ਰਾਫ_compute, ctx->global_ctx);
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਬਣਤਰ ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    ਤਾਰਕਿਕ                  recorded = ਝੂਠ;
};

ਸਥਿਰ ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    ਸਵੈਚਾਲਿਤ * event_ctx      = ਨਵਾਂ ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    ਸਵੈਚਾਲਿਤ * event   = ਨਵਾਂ ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    ਵਾਪਸ event;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਮਿਟਾਓ static_cast<ggml_backend_webgpu_event_context *>(event->context);
    ਮਿਟਾਓ event;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    ਜੇ (!event_ctx->recorded) {
        ਵਾਪਸ;
    }
    wgpu::Wait_ਸਥਿਤੀ ਸਥਿਤੀ =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    ਜੇ (ਸਥਿਤੀ == wgpu::Wait_ਸਥਿਤੀ::Timed_ਬਾਹਰਲਾ) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = ਝੂਠ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_event_ਰਿਕਾਰਡ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.On_Submitted_Work_ਪੂਰਾ(
        wgpu::Callback_ਬਹੁਲਕ::AllowSpontaneous, [](wgpu::ਕਤਾਰ_Work_ਪੂਰਾ_ਸਥਿਤੀ, wgpu::StringView) {});
    event_ctx->recorded = ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ਅਣਵਰਤਿਆ(backend);
    ggml_backend_webgpu_device_event_synchronize(ਖਾਲੀ_ਸੂਚਕ, event);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_ਸੈੱਟ_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ਅਚਲ ਖਾਲੀ *   ਅੰਕੜੇ,
                                                 size_t         ਵਿੱਥ,
                                                 size_t         size) {
    GGML_ਅਣਵਰਤਿਆ(backend);
    ਸਵੈਚਾਲਿਤ * buf_ctx      = (ggml_backend_webgpu_ਬਫਰ_context *) tensor->ਬਫਰ->context;
    size_t ਕੁੱਲ_ਵਿੱਥ = ggml_webgpu_tensor_ਵਿੱਥ(tensor) + ਵਿੱਥ;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_ਬਫਰ(buf_ctx->ਬਫਰ, ਕੁੱਲ_ਵਿੱਥ, ਅੰਕੜੇ, (size / 4) * 4);

    ਜੇ (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t ਬਚਿਆ_ਮਾਪ = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        ਲਈ (size_t i = 0; i < ਬਚਿਆ_ਮਾਪ; i++) {
            ((uint8_t *) &val32)[i] = ((ਅਚਲ uint8_t *) ਅੰਕੜੇ)[size - ਬਚਿਆ_ਮਾਪ + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_ਬਫਰ_memset(buf_ctx->global_ctx, buf_ctx->ਬਫਰ, val32,
                                          ਕੁੱਲ_ਵਿੱਥ + (size - ਬਚਿਆ_ਮਾਪ), ਬਚਿਆ_ਮਾਪ);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_ਕਤਾਰ(backend_ctx->webgpu_ctx->global_ctx);
}

ਸਥਿਰ ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ਨਿਰਾ,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ਨਿਰਾ,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_webgpu_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_ਰਿਕਾਰਡ,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_webgpu_ਬਫਰ_context * ctx = static_cast<ggml_backend_webgpu_ਬਫਰ_context *>(ਬਫਰ->context);
    ਜੇ (ctx != ਖਾਲੀ_ਸੂਚਕ && ctx->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
        ctx->ਬਫਰ.Destroy();
        ਮਿਟਾਓ ctx;
    }
}

// Returns the "fake" base pointer.
ਸਥਿਰ ਖਾਲੀ * ggml_backend_webgpu_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
    ਵਾਪਸ webgpu_ptr_ਅਧਾਰ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                ਵਿੱਥ,
                                                     size_t                size) {
    ਜੇ (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        ਵਾਪਸ;
    }

    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(memset_tensor);

    ggml_backend_webgpu_ਬਫਰ_context * buf_ctx = (ggml_backend_webgpu_ਬਫਰ_context *) ਬਫਰ->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << ਵਿੱਥ << ", " << size << ")");

    size_t ਕੁੱਲ_ਵਿੱਥ = ggml_webgpu_tensor_ਵਿੱਥ(tensor) + ਵਿੱਥ;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_ਬਫਰ_memset(buf_ctx->global_ctx, buf_ctx->ਬਫਰ, val32, ਕੁੱਲ_ਵਿੱਥ, size);
    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(memset_tensor, buf_ctx->global_ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                  ggml_tensor *         tensor,
                                                  ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                  size_t                ਵਿੱਥ,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(ਸੈੱਟ_tensor);
    ggml_backend_webgpu_ਬਫਰ_context * buf_ctx = (ggml_backend_webgpu_ਬਫਰ_context *) ਬਫਰ->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << ਅੰਕੜੇ
                                                              << ", " << ਵਿੱਥ << ", " << size << ")");

    size_t ਕੁੱਲ_ਵਿੱਥ = ggml_webgpu_tensor_ਵਿੱਥ(tensor) + ਵਿੱਥ;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    ਜੇ (ਕੁੱਲ_ਵਿੱਥ % 4 != 0) {
        size_t lane = ਕੁੱਲ_ਵਿੱਥ % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t ਸਿਰ = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t ਸਿਰ_val = 0;
        ਲਈ (size_t i = 0; i < ਸਿਰ; i++) {
            ((uint8_t *) &ਸਿਰ_val)[lane + i] = ((ਅਚਲ uint8_t *) ਅੰਕੜੇ)[i];
        }
        ggml_backend_webgpu_ਬਫਰ_memset(buf_ctx->global_ctx, buf_ctx->ਬਫਰ, ਸਿਰ_val, ਕੁੱਲ_ਵਿੱਥ, ਸਿਰ);

        ਕੁੱਲ_ਵਿੱਥ += ਸਿਰ;
        size -= ਸਿਰ;
        ਅੰਕੜੇ = (ਅਚਲ uint8_t *) ਅੰਕੜੇ + ਸਿਰ;
    }

    ਜੇ (size > 0) {
        buf_ctx->global_ctx->queue.Write_ਬਫਰ(buf_ctx->ਬਫਰ, ਕੁੱਲ_ਵਿੱਥ, ਅੰਕੜੇ, (size / 4) * 4);
    }

    ਜੇ (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t ਬਚਿਆ_ਮਾਪ = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        ਲਈ (size_t i = 0; i < ਬਚਿਆ_ਮਾਪ; i++) {
            ((uint8_t *) &val32)[i] = ((ਅਚਲ uint8_t *) ਅੰਕੜੇ)[size - ਬਚਿਆ_ਮਾਪ + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_ਬਫਰ_memset(buf_ctx->global_ctx, buf_ctx->ਬਫਰ, val32,
                                          ਕੁੱਲ_ਵਿੱਥ + (size - ਬਚਿਆ_ਮਾਪ), ਬਚਿਆ_ਮਾਪ);
    }
    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(ਸੈੱਟ_tensor, buf_ctx->global_ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                  ਅਚਲ ggml_tensor *   tensor,
                                                  ਖਾਲੀ *                ਅੰਕੜੇ,
                                                  size_t                ਵਿੱਥ,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(get_tensor);
    ggml_backend_webgpu_ਬਫਰ_context * buf_ctx = (ggml_backend_webgpu_ਬਫਰ_context *) ਬਫਰ->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << ਅੰਕੜੇ
                                                              << ", " << ਵਿੱਥ << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t ਕੁੱਲ_ਵਿੱਥ = ggml_webgpu_tensor_ਵਿੱਥ(tensor) + ਵਿੱਥ;

    size_t local_ਵਿੱਥ = ਕੁੱਲ_ਵਿੱਥ % 4;
    ਜੇ (local_ਵਿੱਥ != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        ਕੁੱਲ_ਵਿੱਥ -= local_ਵਿੱਥ;
    }

    size_t final_ਮਾਪ = size + local_ਵਿੱਥ;
    ਜੇ (final_ਮਾਪ % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_ਮਾਪ += 4 - (final_ਮਾਪ % 4);
    }

    std::lock_guard<std::recursive_mutex> ਤਾਲਾ(buf_ctx->global_ctx->mutex);

    ਜੇ (buf_ctx->global_ctx->get_tensor_staging_buf == ਖਾਲੀ_ਸੂਚਕ ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_ਮਾਪ() < final_ਮਾਪ) {
        // Create a new staging buffer if it doesn't exist or is too small
        ਜੇ (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_ਬਫਰ(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_ਮਾਪ,
                                  wgpu::ਬਫਰ_Usage::CopyDst | wgpu::ਬਫਰ_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_ਬਫਰ_ਮੰਜ਼ਿਲ_ਬਫਰ(buf_ctx->ਬਫਰ, ਕੁੱਲ_ਵਿੱਥ, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_ਮਾਪ);
    wgpu::Command_ਬਫਰ commands = encoder.ਅੰਤ_2();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_ਬਫਰ(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::Map_ਬਹੁਲਕ::Read, 0, final_ਮਾਪ);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ਅਚਲ ਖਾਲੀ * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_ਮਾਪ);

    // Copy the data from the mapped range to the output buffer
    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਖਾਲੀ *) ((ਅਚਲ ਅੱਖਰ *) mapped_range + local_ਵਿੱਥ), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(get_tensor, buf_ctx->global_ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << ਬਫਰ << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(clear);
    ggml_backend_webgpu_ਬਫਰ_context * buf_ctx = (ggml_backend_webgpu_ਬਫਰ_context *) ਬਫਰ->context;
    ggml_backend_webgpu_ਬਫਰ_memset(buf_ctx->global_ctx, buf_ctx->ਬਫਰ, value, 0, ਬਫਰ->size);
    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(clear, buf_ctx->global_ctx);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_webgpu_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_webgpu_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ਨਿਰਾ,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ਨਿਰਾ,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_webgpu_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ਵਾਪਸ ctx->device_name.c_str();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_webgpu_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft,
                                                                          size_t                     size) {
    ਸਥਿਰ std::atomic<ਪੂਰਨ> ਬਫਰ_ਗਿਣਤੀ;
    ਪੂਰਨ                     ਬਫਰ_id = ਬਫਰ_ਗਿਣਤੀ++;
    std::string             buf_name  = "tensor_buf" + std::to_string(ਬਫਰ_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << ਬਫਰ_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::ਬਫਰ_2                         buf;
    ggml_webgpu_create_ਬਫਰ(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::ਬਫਰ_Usage::Storage | wgpu::ਬਫਰ_Usage::CopySrc | wgpu::ਬਫਰ_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_ਬਫਰ_context * buf_ctx =
        ਨਵਾਂ ggml_backend_webgpu_ਬਫਰ_context(buf, buf_name, ctx->webgpu_global_ctx);

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_webgpu_ਬਫਰ_interface, buf_ctx, size);
}

ਸਥਿਰ size_t ggml_backend_webgpu_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ਵਾਪਸ dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
ਸਥਿਰ size_t ggml_backend_webgpu_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ਵਾਪਸ dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ;
}

ਸਥਿਰ size_t ggml_backend_webgpu_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft,
                                                             ਅਚਲ ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    ਚੋਣ (tensor->op) {
        ਹਾਲਤ GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            {
                ਅਚਲ ggml_tensor * src0 = tensor->src[0];
                ਜੇ (src0) {
                    ਅਚਲ size_t ਭਰਿਆ = ਆਕਾਰ(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        ਭਰਿਆ * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                ਅਚਲ ggml_tensor * Q            = tensor->src[0];
                ਅਚਲ ggml_tensor * K            = tensor->src[1];
                ਅਚਲ ggml_tensor * V            = tensor->src[2];
                ਅਚਲ ggml_tensor * ਨਕਾਬ         = tensor->src[3];
                ਅਚਲ ਸਵੈਚਾਲਿਤ &        capabilities = ctx->webgpu_global_ctx->capabilities;
                ਜੇ (ggml_webgpu_flash_attn_use_vec_ਰਸਤਾ(ctx->webgpu_global_ctx, Q, K, V)) {
                    ਅਚਲ ਤਾਰਕਿਕ kv_ਸਿੱਧਾ =
                        ggml_webgpu_flash_attn_k_ਸਿੱਧਾ(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ) ||
                        ggml_webgpu_flash_attn_v_ਸਿੱਧਾ(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ);
                    ਅਚਲ uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_ਮਾਪ, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ, kv_ਸਿੱਧਾ);

                    ਅਚਲ uint32_t vec_nwg_cap = capabilities.min_subgroup_ਮਾਪ;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ਅਚਲ size_t   align = capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;
                    ਅਚਲ uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    ਜੇ (nwg > 1u) {
                        ਅਚਲ uint64_t tmp_ਅੰਕੜੇ_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ਅਚਲ uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ਅਚਲ size_t   tmp_ਮਾਪ_ਬਾਈਟਾਂ = ROUNDUP_POW2((tmp_ਅੰਕੜੇ_elems + tmp_stats_elems) * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_ਮਾਪ_ਬਾਈਟਾਂ + align;
                    } ਨਹੀਂਤਾਂ {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    ਜੇ (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ) {
                        ਅਚਲ uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ਅਚਲ uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ਅਚਲ uint32_t stride_mask3    = (uint32_t) (ਨਕਾਬ->nb[3] / ggml_type_ਮਾਪ(ਨਕਾਬ->type));
                        ਅਚਲ uint32_t blk_batch_ਗਿਣਤੀ = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ਅਚਲ uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_ਗਿਣਤੀ;
                        ਅਚਲ size_t   blk_ਮਾਪ_ਬਾਈਟਾਂ =
                            ROUNDUP_POW2(blk_elems * ਆਕਾਰ(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_ਮਾਪ_ਬਾਈਟਾਂ + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                ਅਚਲ ggml_tensor * src0 = tensor->src[0];
                ਅਚਲ ggml_tensor * src1 = tensor->src[1];
                ਤਾਰਕਿਕ                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_ਗੁਣਨਫਲ,
                                             ctx->webgpu_global_ctx->vendor);
                ਜੇ (use_mmvq) {
                    ਅਚਲ size_t q8_src1_ਮਾਪ = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_ਮਾਪ +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                ਅਚਲ ggml_tensor * src0 = tensor->src[0];
                ਅਚਲ ggml_tensor * src1 = tensor->src[1];
                ਜੇ (src0 && src1) {
                    ਅਚਲ size_t ids_cache_ਮਾਪ = ਆਕਾਰ(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    ਅਚਲ size_t ids_tiles_ਮਾਪ = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_ਮਾਪ + ids_tiles_ਮਾਪ +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }
    ਵਾਪਸ res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    ਵਾਪਸ ctx->device_name.c_str();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    ਵਾਪਸ ctx->device_desc.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_ਬਫਰ_ਮਾਪ = ctx->webgpu_global_ctx->capabilities.limits.max_ਬਫਰ_ਮਾਪ_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#ਜੇ UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_ਮਾਪ = static_cast<uint64_t>(UINTPTR_MAX);
    ਜੇ (max_ਬਫਰ_ਮਾਪ > max_ptr_ਮਾਪ) {
        max_ਬਫਰ_ਮਾਪ = max_ptr_ਮਾਪ;
    }
#ਜੇ_ਅੰਤ
    *ਮੁਕਤ  = static_cast<size_t>(max_ਬਫਰ_ਮਾਪ);
    *ਕੁੱਲ = static_cast<size_t>(max_ਬਫਰ_ਮਾਪ);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                 = */ ਝੂਠ,
        /* .host_buffer           = */ ਝੂਠ,
        /* .buffer_from_host_ptr  = */ ਝੂਠ,
        /* .events                = */ ਝੂਠ,
        /* .mmap_support          = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_guid_t ggml_backend_webgpu_guid(ਖਾਲੀ) {
    ਸਥਿਰ ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    ਵਾਪਸ &guid;
}

ਸਥਿਰ ਖਾਲੀ ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_ਧਾਗੇ = ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup *
                         ctx->capabilities.limits.max_Compute_Workgroups_ਪ੍ਰਤੀ_Dimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ, max_ਧਾਗੇ);
    std::vector<wgpu::Constant_ਇੰਦਰਾਜ> constants(2);
    constants[0].ਕੁੰਜੀ     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
    constants[1].ਕੁੰਜੀ     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#ਜੇ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ਅਚਲ ਅੱਖਰ * ਅਚਲ          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_ਗਿਣਤੀ = 2;
    options.ਅਗਲਾ_ਵਿੱਚ_Chain                   = &adapterTogglesDesc;
#ਜੇ_ਅੰਤ

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::Callback_ਬਹੁਲਕ::AllowSpontaneous,
            [&adapter](wgpu::Request_Adapter_ਸਥਿਤੀ ਸਥਿਤੀ, wgpu::Adapter _adapter, wgpu::StringView ਸੁਨੇਹਾ) {
                ਜੇ (ਸਥਿਤੀ != wgpu::Request_Adapter_ਸਥਿਤੀ::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(ਸੁਨੇਹਾ).c_str());
                    ਵਾਪਸ;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

ਸਥਿਰ ਖਾਲੀ create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != ਖਾਲੀ_ਸੂਚਕ);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#ਜੇ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::Adapter_Properties_Subgroup_ਮੈਟ੍ਰਿਕਸ_Configs subgroup_ਮੈਟ੍ਰਿਕਸ_configs{};
    ਜੇ (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ਮੈਟ੍ਰਿਕਸ)) {
        info.ਅਗਲਾ_ਵਿੱਚ_Chain = &subgroup_ਮੈਟ੍ਰਿਕਸ_configs;
    }
#ਜੇ_ਅੰਤ
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_ਮਾਪ = ggml_backend_webgpu_get_command_submit_batch_ਮਾਪ();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_ਗੁਣਨਫਲ = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8_Integer_Dot_ਗੁਣਨਫਲ);

    ਤਾਰਕਿਕ valid_subgroup_ਮੈਟ੍ਰਿਕਸ_config = ਝੂਠ;
#ਜੇ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    ਜੇ (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ਮੈਟ੍ਰਿਕਸ)) {
        ਲਈ (size_t i = 0; i < subgroup_ਮੈਟ੍ਰਿਕਸ_configs.config_ਗਿਣਤੀ; i++) {
            ਅਚਲ wgpu::Subgroup_ਮੈਟ੍ਰਿਕਸ_Config config = subgroup_ਮੈਟ੍ਰਿਕਸ_configs.configs[i];
            ਜੇ (config.componentType == wgpu::Subgroup_ਮੈਟ੍ਰਿਕਸ_Component_Type::F16 &&
                config.ਨਤੀਜਾ_Component_Type == wgpu::Subgroup_ਮੈਟ੍ਰਿਕਸ_Component_Type::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_ਮੈਟ੍ਰਿਕਸ_config                  = ਸੱਚ;
                ਰੋਕੋ;
            }
        }
    }
#ਜੇ_ਅੰਤ
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_ਮੈਟ੍ਰਿਕਸ = valid_subgroup_ਮੈਟ੍ਰਿਕਸ_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_ਮਾਪ = info.subgroup_Min_ਮਾਪ;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_ਮਾਪ = info.subgroup_Max_ਮਾਪ;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#ਜੇ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    ਜੇ (ctx->webgpu_global_ctx->capabilities.supports_subgroup_ਮੈਟ੍ਰਿਕਸ) {
        required_features.push_back(wgpu::FeatureName::Chromium_Experimental_Subgroup_ਮੈਟ੍ਰਿਕਸ);
    }
#ਜੇ_ਅੰਤ

    ਜੇ (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#ਜੇ_ਅੰਤ

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.ਅੰਕੜੇ();
    dev_desc.required_Feature_ਗਿਣਤੀ = required_features.size();
    dev_desc.ਸੈੱਟ_Device_ਗੁਆਚਾ_Callback(
        wgpu::Callback_ਬਹੁਲਕ::AllowSpontaneous,
        [](ਅਚਲ wgpu::Device & device, wgpu::Device_ਗੁਆਚਾ_Reason reason, wgpu::StringView ਸੁਨੇਹਾ) {
            ਜੇ (reason == wgpu::Device_ਗੁਆਚਾ_Reason::Destroyed) {
                ਵਾਪਸ;
            }
            GGML_ਅਣਵਰਤਿਆ(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<ਪੂਰਨ>(reason),
                           std::string(ਸੁਨੇਹਾ).c_str());
        });
    dev_desc.ਸੈੱਟ_Uncaptured_Error_Callback(
        [](ਅਚਲ wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView ਸੁਨੇਹਾ) {
            GGML_ਅਣਵਰਤਿਆ(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<ਪੂਰਨ>(reason),
                       std::string(ਸੁਨੇਹਾ).c_str());
        });

#ਜੇ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    ਅਚਲ ਅੱਖਰ * ਅਚਲ          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    ਅਚਲ ਅੱਖਰ * ਅਚਲ          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_ਗਿਣਤੀ  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_ਗਿਣਤੀ = 1;

    dev_desc.ਅਗਲਾ_ਵਿੱਚ_Chain = &deviceTogglesDesc;
#ਜੇ_ਅੰਤ

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::Callback_ਬਹੁਲਕ::AllowSpontaneous,
            [ctx](wgpu::Request_Device_ਸਥਿਤੀ ਸਥਿਤੀ, wgpu::Device device, wgpu::StringView ਸੁਨੇਹਾ) {
                ਜੇ (ਸਥਿਤੀ != wgpu::Request_Device_ਸਥਿਤੀ::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(ਸੁਨੇਹਾ).c_str());
                    ਵਾਪਸ;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != ਖਾਲੀ_ਸੂਚਕ);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_ਬਫਰ(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_ਮਾਪ_ਬਾਈਟਾਂ, wgpu::ਬਫਰ_Usage::CopyDst | wgpu::ਬਫਰ_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.Get_ਕਤਾਰ();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

ਸਥਿਰ webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_ਮਾਪ_ਬਾਈਟਾਂ,
        webgpu_ctx->global_ctx->command_submit_batch_ਮਾਪ + WEBGPU_ਅੰਕ_PARAM_ਥਾਂ_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_ਇਕਸਾਰ_ਬਫਰ_ਵਿੱਥ_Alignment);
    ggml_webgpu_create_ਬਫਰ(webgpu_ctx->global_ctx->device, webgpu_ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_dev_error_buf,
                              WEBGPU_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ERROR_BUF_ਮਾਪ_ਬਾਈਟਾਂ,
                              wgpu::ਬਫਰ_Usage::Storage | wgpu::ਬਫਰ_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_ਬਫਰ(webgpu_ctx->global_ctx->device, webgpu_ctx->ਸੈੱਟ_ਪੰਕਤੀਆਂ_host_error_buf,
                              WEBGPU_ਸੈੱਟ_ਪੰਕਤੀਆਂ_ERROR_BUF_ਮਾਪ_ਬਾਈਟਾਂ,
                              wgpu::ਬਫਰ_Usage::CopyDst | wgpu::ਬਫਰ_Usage::MapRead, "set_rows_host_error_buf");

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = ਝੂਠ;
    ggml_webgpu_create_ਬਫਰ(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_ਮਾਪ_ਬਾਈਟਾਂ,
        wgpu::ਬਫਰ_Usage::QueryResolve | wgpu::ਬਫਰ_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_ਬਫਰ(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_ਮਾਪ_ਬਾਈਟਾਂ,
                              wgpu::ਬਫਰ_Usage::CopyDst | wgpu::ਬਫਰ_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::Query_ਸੈੱਟ_Descriptor query_ਸੈੱਟ_desc = {};
    query_ਸੈੱਟ_desc.type                     = wgpu::QueryType::Timestamp;
    query_ਸੈੱਟ_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_ਗਿਣਤੀ;
    webgpu_ctx->profile_timestamp_query_ਸੈੱਟ = webgpu_ctx->global_ctx->device.Create_Query_ਸੈੱਟ(&query_ਸੈੱਟ_desc);
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_ਬਫਰ(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * ਆਕਾਰ(uint32_t),
                              wgpu::ਬਫਰ_Usage::CopyDst | wgpu::ਬਫਰ_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_ਬਫਰ(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * ਆਕਾਰ(uint32_t),
                              wgpu::ਬਫਰ_Usage::Storage | wgpu::ਬਫਰ_Usage::CopySrc, "debug_dev_buf");
#ਜੇ_ਅੰਤ
    ਵਾਪਸ webgpu_ctx;
}

ਸਥਿਰ ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    GGML_ਅਣਵਰਤਿਆ(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ਸਵੈਚਾਲਿਤ * backend_ctx      = ਨਵਾਂ ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    ਸਵੈਚਾਲਿਤ * backend = ਨਵਾਂ ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    ਵਾਪਸ backend;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_webgpu_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_webgpu_ਬਫਰ_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_ਬਫਰ_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_ਬਫਰ_type_alloc_ਬਫਰ,
                        /* .alloc_buffer_n      = */ ਨਿਰਾ,
                        /* .get_alignment       = */ ggml_backend_webgpu_ਬਫਰ_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_ਬਫਰ_type_get_max_ਮਾਪ,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_ਬਫਰ_type_get_alloc_ਮਾਪ,
                        /* .get_alloc_size_n    = */ ਨਿਰਾ,
                        /* .is_host             = */ ਨਿਰਾ,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ ਨਿਰਾ
    };

    ਵਾਪਸ &ggml_backend_webgpu_ਬਫਰ_type;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ buft->iface.get_name == ggml_backend_webgpu_ਬਫਰ_type_get_name;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_webgpu_supported_qtype(ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    ਜੇ (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ ||
        (src0 != ਖਾਲੀ_ਸੂਚਕ &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ) ||
        (src1 != ਖਾਲੀ_ਸੂਚਕ &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ supports_op = ਝੂਠ;
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_RESHAPE:
            supports_op = ਸੱਚ;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_CPY:
        ਹਾਲਤ GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ਜੇ (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                ਚੋਣ (src1->type) {
                    ਹਾਲਤ GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        ਰੋਕੋ;
                    ਹਾਲਤ GGML_TYPE_F32:
                        ਚੋਣ (src0->type) {
                            ਹਾਲਤ GGML_TYPE_F32:
                            ਹਾਲਤ GGML_TYPE_F16:
                            ਹਾਲਤ GGML_TYPE_BF16:
                            ਹਾਲਤ GGML_TYPE_Q1_0:
                            ਹਾਲਤ GGML_TYPE_Q2_0:
                            ਹਾਲਤ GGML_TYPE_Q4_0:
                            ਹਾਲਤ GGML_TYPE_Q4_1:
                            ਹਾਲਤ GGML_TYPE_Q5_0:
                            ਹਾਲਤ GGML_TYPE_Q5_1:
                            ਹਾਲਤ GGML_TYPE_Q8_0:
                            ਹਾਲਤ GGML_TYPE_Q2_K:
                            ਹਾਲਤ GGML_TYPE_Q3_K:
                            ਹਾਲਤ GGML_TYPE_Q4_K:
                            ਹਾਲਤ GGML_TYPE_Q5_K:
                            ਹਾਲਤ GGML_TYPE_Q6_K:
                            ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                            ਹਾਲਤ GGML_TYPE_IQ2_XS:
                            ਹਾਲਤ GGML_TYPE_IQ2_S:
                            ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                            ਹਾਲਤ GGML_TYPE_IQ3_S:
                            ਹਾਲਤ GGML_TYPE_IQ1_S:
                            ਹਾਲਤ GGML_TYPE_IQ1_M:
                            ਹਾਲਤ GGML_TYPE_IQ4_NL:
                            ਹਾਲਤ GGML_TYPE_IQ4_XS:
                            ਹਾਲਤ GGML_TYPE_MXFP4:
                            ਹਾਲਤ GGML_TYPE_NVFP4:
                                supports_op = ਸੱਚ;
                                ਰੋਕੋ;
                            ਮੂਲ:
                                ਰੋਕੋ;
                        }
                    ਮੂਲ:
                        ਰੋਕੋ;
                }
                ਰੋਕੋ;
            }
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            ਚੋਣ (src1->type) {
                ਹਾਲਤ GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_TYPE_F32:
                    ਚੋਣ (src0->type) {
                        ਹਾਲਤ GGML_TYPE_F32:
                        ਹਾਲਤ GGML_TYPE_F16:
                        ਹਾਲਤ GGML_TYPE_BF16:
                        ਹਾਲਤ GGML_TYPE_Q1_0:
                        ਹਾਲਤ GGML_TYPE_Q4_0:
                        ਹਾਲਤ GGML_TYPE_Q4_1:
                        ਹਾਲਤ GGML_TYPE_Q5_0:
                        ਹਾਲਤ GGML_TYPE_Q5_1:
                        ਹਾਲਤ GGML_TYPE_Q8_0:
                        ਹਾਲਤ GGML_TYPE_Q2_K:
                        ਹਾਲਤ GGML_TYPE_Q3_K:
                        ਹਾਲਤ GGML_TYPE_Q4_K:
                        ਹਾਲਤ GGML_TYPE_Q5_K:
                        ਹਾਲਤ GGML_TYPE_Q6_K:
                        ਹਾਲਤ GGML_TYPE_IQ1_S:
                        ਹਾਲਤ GGML_TYPE_IQ1_M:
                        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ2_XS:
                        ਹਾਲਤ GGML_TYPE_IQ2_S:
                        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                        ਹਾਲਤ GGML_TYPE_IQ3_S:
                        ਹਾਲਤ GGML_TYPE_IQ4_NL:
                        ਹਾਲਤ GGML_TYPE_IQ4_XS:
                        ਹਾਲਤ GGML_TYPE_MXFP4:
                        ਹਾਲਤ GGML_TYPE_NVFP4:
                            supports_op = ਸੱਚ;
                            ਰੋਕੋ;
                        ਮੂਲ:
                            ਰੋਕੋ;
                    }
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਰੋਕੋ;
            }
            ਜੇ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                ਜੇ (!supports_op) {
                    ਰੋਕੋ;
                }
                ਜੇ (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = ਝੂਠ;
                    ਰੋਕੋ;
                }
                ਅਚਲ ਸਵੈਚਾਲਿਤ & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ਅਚਲ size_t storage_ਵਿੱਥ_alignment = capabilities.limits.min_Storage_ਬਫਰ_ਵਿੱਥ_Alignment;

                // subgroup matrix path requirements
                ਅਚਲ ਤਾਰਕਿਕ use_subgroup_ਮੈਟ੍ਰਿਕਸ = ggml_webgpu_flash_attn_can_use_subgroup_ਮੈਟ੍ਰਿਕਸ_ਰਸਤਾ(
                    capabilities.supports_subgroup_ਮੈਟ੍ਰਿਕਸ, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ਅਚਲ ਤਾਰਕਿਕ float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_ਵਿੱਥ_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_ਵਿੱਥ_alignment));
                ਅਚਲ uint32_t k_tile_ਸਿਰ_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ :
                                                       (uint32_t) ggml_blck_ਮਾਪ(src1->type);
                ਅਚਲ uint32_t v_tile_ਸਿਰ_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ਚੌੜਾਈ :
                                                       (uint32_t) ggml_blck_ਮਾਪ(src2->type);
                ਅਚਲ ਤਾਰਕਿਕ     tile_kv_ਸਿਰ_dims_aligned =
                    src0->ne[0] % k_tile_ਸਿਰ_align == 0 && src2->ne[0] % v_tile_ਸਿਰ_align == 0;
                ਅਚਲ ਤਾਰਕਿਕ tile_can_dispatch_all_q_ਪੰਕਤੀਆਂ =
                    capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_ਮਾਪ;
                ਅਚਲ ਤਾਰਕਿਕ use_tile = !use_subgroup_ਮੈਟ੍ਰਿਕਸ && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_ਸਿਰ_dims_aligned && tile_can_dispatch_all_q_ਪੰਕਤੀਆਂ;

                ਜੇ (!use_subgroup_ਮੈਟ੍ਰਿਕਸ && !use_tile) {
                    supports_op = ਝੂਠ;
                    ਰੋਕੋ;
                }
                ਅਚਲ uint32_t q_tile =
                    use_subgroup_ਮੈਟ੍ਰਿਕਸ ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ਅਚਲ uint32_t kv_granularity = use_subgroup_ਮੈਟ੍ਰਿਕਸ ? capabilities.sg_mat_n : 1u;
                ਅਚਲ ਤਾਰਕਿਕ     kv_ਸਿੱਧਾ = use_subgroup_ਮੈਟ੍ਰਿਕਸ ?
                                               ggml_webgpu_flash_attn_k_ਸਿੱਧਾ(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_ਸਿੱਧਾ(src0, src2, capabilities.sg_mat_k) :
                                               ਝੂਠ;
                ਅਚਲ uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_ਮਾਪ, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != ਖਾਲੀ_ਸੂਚਕ, kv_ਸਿੱਧਾ);
                supports_op = max_kv_tile > 0;
                ਰੋਕੋ;
            }
        ਹਾਲਤ GGML_OP_RMS_NORM:
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(op)) {
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਰੋਕੋ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            {
                ਅਚਲ ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                ਚੋਣ (UNARY_OP) {
                    ਹਾਲਤ GGML_UNARY_OP_ABS:
                    ਹਾਲਤ GGML_UNARY_OP_SGN:
                    ਹਾਲਤ GGML_UNARY_OP_NEG:
                    ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                    ਹਾਲਤ GGML_UNARY_OP_TANH:
                    ਹਾਲਤ GGML_UNARY_OP_ELU:
                    ਹਾਲਤ GGML_UNARY_OP_RELU:
                    ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                    ਹਾਲਤ GGML_UNARY_OP_GELU:
                    ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                    ਹਾਲਤ GGML_UNARY_OP_SILU:
                    ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                    ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                    ਹਾਲਤ GGML_UNARY_OP_EXP:
                    ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                    ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                    ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                    ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                    ਹਾਲਤ GGML_UNARY_OP_CEIL:
                    ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                    ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                    ਹਾਲਤ GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        ਰੋਕੋ;
                    ਮੂਲ:
                        ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            {
                ਅਚਲ uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_ਪ੍ਰਤੀ_Workgroup;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }
    ਜੇ (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ ||
        (src0 != ਖਾਲੀ_ਸੂਚਕ &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ) ||
        (src1 != ਖਾਲੀ_ਸੂਚਕ &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ) ||
        (src2 != ਖਾਲੀ_ਸੂਚਕ &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ਬਫਰ_Binding_ਮਾਪ)) {
        supports_op = ਝੂਠ;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    ਜੇ (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } ਨਹੀਂਤਾਂ {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    ਵਾਪਸ supports_op;
}

ਸਥਿਰ ਬਣਤਰ ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr = */ ਨਿਰਾ,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ ਨਿਰਾ,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    ਵਾਪਸ ctx->name;
}

ਸਥਿਰ size_t ggml_backend_webgpu_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    ਵਾਪਸ ctx->device_ਗਿਣਤੀ;
}

// Only one device is supported for now
ਸਥਿਰ ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ਸੂਚਕਾਂਕ == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸ਼ੁਰੂ(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    ਸਥਿਰ ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    ਸਥਿਰ ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_ਕੁੱਲ_ਸਿਰਾ(reg_get_device, reg_ctx->webgpu_global_ctx);
    ਵਾਪਸ &device;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ ਨਿਰਾ,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    ਸਥਿਰ ggml_backend_webgpu_reg_context * ctx = ਨਵਾਂ ggml_backend_webgpu_reg_context();

    ਸਥਿਰ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_ਗਿਣਤੀ = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    ਜੇ (ctx->webgpu_global_ctx != ਖਾਲੀ_ਸੂਚਕ && ctx->webgpu_global_ctx->instance != ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.ਅੰਕੜੇ();
    instance_descriptor.required_Feature_ਗਿਣਤੀ                 = instance_features.size();

#ਜੇ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ਅਚਲ ਅੱਖਰ * ਅਚਲ          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_ਗਿਣਤੀ = 1;
    instance_descriptor.ਅਗਲਾ_ਵਿੱਚ_Chain        = &instanceTogglesDesc;
#ਜੇ_ਅੰਤ

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(ਨਵਾਂ webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    ਜੇ (ctx->webgpu_global_ctx->instance != ਖਾਲੀ_ਸੂਚਕ) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    ਜੇ (adapter != ਖਾਲੀ_ਸੂਚਕ && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#ਜੇ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#ਜੇ_ਅੰਤ
    ) {
        ctx->device_ਗਿਣਤੀ = 1;
    }

    ਵਾਪਸ &reg;
}

ggml_backend_t ggml_backend_webgpu_init(ਖਾਲੀ) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    ਜੇ (ggml_backend_reg_dev_ਗਿਣਤੀ(reg) == 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ਵਾਪਸ ggml_backend_webgpu_backend_init(dev, ਖਾਲੀ_ਸੂਚਕ);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
