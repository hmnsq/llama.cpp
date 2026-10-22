/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#ಅನ್ತರ್ಭಾವ "ggml-webgpu.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-webgpu-shader-lib.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <webgpu/webgpu_cpp.h>

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstring>
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <utility>
#ಅನ್ತರ್ಭಾವ <vector>

#ವ್ಯಾಖ್ಯಾನಿಸು ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#ವ್ಯಾಖ್ಯಾನಿಸು CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ compute_2d_workgroups(uint32_t ಒಟ್ಟು_wg, uint32_t max_ಪ್ರತಿ_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(ಒಟ್ಟು_wg, max_ಪ್ರತಿ_dim));
    wg_x = CEIL_DIV(ಒಟ್ಟು_wg, wg_y);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t ggml_webgpu_u32_from_f32(ದಶಮಾಂಶ value) {
    uint32_t ಬಿಟ್ಟುಗಳು;
    ಸ್ಮರಣೆ_ನಕಲು(&ಬಿಟ್ಟುಗಳು, &value, ಗಾತ್ರ(ಬಿಟ್ಟುಗಳು));
    ಹಿಂತಿರುಗಿ ಬಿಟ್ಟುಗಳು;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(ಸಂದೇಶ_2)  std::ಹೊರಗೆ << ಸಂದೇಶ_2 << std::ಸಾಲು_ಅಂತ್ಯ
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#ಇಲ್ಲದಿದ್ದರೆ
#    define WEBGPU_LOG_DEBUG(ಸಂದೇಶ_2) ((ಶೂನ್ಯ) 0)
#ಆದರೆ_ಅಂತ್ಯ  // GGML_WEBGPU_DEBUG

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(id) ಸ್ವಯಂಚಾಲಿತ cpu_ಒಟ್ಟು_ಆರಂಭ##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(id, ctx)                                                         \
        ಸ್ವಯಂಚಾಲಿತ   cpu_ಒಟ್ಟು_ಕೊನೆ##id = std::chrono::high_resolution_clock::now();                            \
        ದ್ವಿಗುಣ cpu_ಒಟ್ಟು_ಸಮಯ##id =                                                                      \
            std::chrono::duration<ದ್ವಿಗುಣ, std::milli>(cpu_ಒಟ್ಟು_ಕೊನೆ##id - cpu_ಒಟ್ಟು_ಆರಂಭ##id).count(); \
        (ctx)->cpu_ಸಮಯ_ms[#id] += cpu_ಒಟ್ಟು_ಸಮಯ##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_ಆರಂಭ(id) ಸ್ವಯಂಚಾಲಿತ cpu_detail_ಆರಂಭ##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_ಕೊನೆ(id, ctx)                                                          \
        ಸ್ವಯಂಚಾಲಿತ   cpu_detail_ಕೊನೆ##id = std::chrono::high_resolution_clock::now();                             \
        ದ್ವಿಗುಣ cpu_detail_ಸಮಯ##id =                                                                       \
            std::chrono::duration<ದ್ವಿಗುಣ, std::milli>(cpu_detail_ಕೊನೆ##id - cpu_detail_ಆರಂಭ##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_ಸಮಯ##id;
#ಇಲ್ಲದಿದ್ದರೆ
#    define WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(id)
#    define WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_ಆರಂಭ(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_ಕೊನೆ(id, ctx)
#ಆದರೆ_ಅಂತ್ಯ  // GGML_WEBGPU_CPU_PROFILE

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_ಎಣಿಕೆ        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು (WEBGPU_MAX_PROFILE_QUERY_ಎಣಿಕೆ * ಗಾತ್ರ(uint64_t))
#ಆದರೆ_ಅಂತ್ಯ

/* Constants */

#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_ಗಾತ್ರ 64u
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_ಅಂಕೆ_PARAM_ಸ್ಥಾನ_SAFETY_MARGIN      10u
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_PARAMS_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು             128  // enough for 32 parameters
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ERROR_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು     4
#ವ್ಯಾಖ್ಯಾನಿಸು WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
ಸ್ಥಿರ ಶೂನ್ಯ * ಸ್ಥಿರಾಂಕ webgpu_ptr_ತಳ = (ಶೂನ್ಯ *) (uintptr_t) 0x1000;  // NOLINT

ಸ್ಥಿರ size_t ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ_tensor = tensor->view_src ? tensor->view_src : tensor;
    ಹಿಂತಿರುಗಿ (size_t) ((uintptr_t) ತಳ_tensor->ದತ್ತಾಂಶ - (uintptr_t) webgpu_ptr_ತಳ) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
ಸ್ಥಿರ ಶೂನ್ಯ ggml_webgpu_create_ಬಫರ್(wgpu::Device &    device,
                                      wgpu::ಬಫರ್_2 &    ಬಫರ್,
                                      size_t            size,
                                      wgpu::ಬಫರ್_Usage usage,
                                      ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
ರಚನೆ webgpu_param_arena {
    wgpu::ಬಫರ್_2 ಬಫರ್;
    size_t       ಸ್ಥಾನ_stride = 0;
    size_t       ಸ್ಥಾನ_ಗಾತ್ರ   = 0;
    uint32_t     ಸ್ಥಾನ_ಎಣಿಕೆ  = 0;
    uint32_t     ಮುಂದಿನ_ಸ್ಥಾನ   = 0;

    ಶೂನ್ಯ init(wgpu::Device device, size_t ಸ್ಥಾನ_ಗಾತ್ರ, uint32_t ಸ್ಥಾನ_ಎಣಿಕೆ, size_t alignment) {
        ಇದು->ಸ್ಥಾನ_stride = ROUNDUP_POW2(ಸ್ಥಾನ_ಗಾತ್ರ, alignment);
        ಇದು->ಸ್ಥಾನ_ಗಾತ್ರ   = ಸ್ಥಾನ_ಗಾತ್ರ;
        ಇದು->ಸ್ಥಾನ_ಎಣಿಕೆ  = ಸ್ಥಾನ_ಎಣಿಕೆ;
        ಇದು->ಮುಂದಿನ_ಸ್ಥಾನ   = 0;

        ggml_webgpu_create_ಬಫರ್(device, ಬಫರ್, ಇದು->ಸ್ಥಾನ_stride * ಸ್ಥಾನ_ಎಣಿಕೆ,
                                  wgpu::ಬಫರ್_Usage::CopyDst | wgpu::ಬಫರ್_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_ಸ್ಥಾನ(size_t size) {
        GGML_ASSERT(size <= ಸ್ಥಾನ_ಗಾತ್ರ);
        ಆದರೆ (ಮುಂದಿನ_ಸ್ಥಾನ >= ಸ್ಥಾನ_ಎಣಿಕೆ) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        ಹಿಂತಿರುಗಿ ಸ್ಥಾನ_stride * ಮುಂದಿನ_ಸ್ಥಾನ++;
    }

    ಶೂನ್ಯ reset() { ಮುಂದಿನ_ಸ್ಥಾನ = 0; }

    ಶೂನ್ಯ cleanup() {
        ಆದರೆ (ಬಫರ್) {
            ಬಫರ್.Destroy();
            ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
        }
    }

    ~webgpu_param_arena() { ಇದು->cleanup(); }
};

ರಚನೆ webgpu_encoded_op {
    uint32_t ಅಂಕೆ_kernels = 0;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_ಹೆಸರುಗಳು;
#ಆದರೆ_ಅಂತ್ಯ
};

ರಚನೆ webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> bind_ಗುಂಪು_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

ರಚನೆ webgpu_capabilities {
    wgpu::Limits limits;
    ತಾರ್ಕಿಕ         supports_subgroups       = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ         supports_subgroup_ಮಾತೃಕೆ = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ         supports_dot_ಗುಣಲಬ್ಧ     = ಸುಳ್ಳು;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_ಗಾತ್ರ     = 0;
    uint32_t min_subgroup_ಗಾತ್ರ = 0;
    uint32_t max_subgroup_ಗಾತ್ರ = 0;
    size_t   memset_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
};

// Stores global webgpu members
ರಚನೆ webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::ಸರತಿ_2    queue;
    uint32_t       command_submit_batch_ಗಾತ್ರ = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_ಗಾತ್ರ;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::ಬಫರ್_2         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::ಬಫರ್_2    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, ದ್ವಿಗುಣ> cpu_ಸಮಯ_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, ದ್ವಿಗುಣ> cpu_detail_ms;
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_DEBUG
    wgpu::ಬಫರ್_2 debug_host_buf;
    wgpu::ಬಫರ್_2 debug_dev_buf;
#ಆದರೆ_ಅಂತ್ಯ

    ~webgpu_global_context_struct() {
        ಆದರೆ (ಇದು->get_tensor_staging_buf) {
            ಇದು->get_tensor_staging_buf.Destroy();
            ಇದು->get_tensor_staging_buf = ಶೂನ್ಯಸೂಚಕ;
        }
        ಆದರೆ (ಇದು->memset_params_buf) {
            ಇದು->memset_params_buf.Destroy();
            ಇದು->memset_params_buf = ಶೂನ್ಯಸೂಚಕ;
        }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_DEBUG
        ಆದರೆ (ಇದು->debug_host_buf) {
            ಇದು->debug_host_buf.Destroy();
            ಇದು->debug_host_buf = ಶೂನ್ಯಸೂಚಕ;
        }
        ಆದರೆ (ಇದು->debug_dev_buf) {
            ಇದು->debug_dev_buf.Destroy();
            ಇದು->debug_dev_buf = ಶೂನ್ಯಸೂಚಕ;
        }
#ಆದರೆ_ಅಂತ್ಯ
    }
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
ರಚನೆ webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::ಬಫರ್_2             ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf;
    wgpu::ಬಫರ್_2             ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf;
    wgpu::CommandEncoder     ಸಕ್ರಿಯ_command_encoder;
    wgpu::ComputePassEncoder ಸಕ್ರಿಯ_compute_pass;
    ತಾರ್ಕಿಕ                     batch_compute_passes = ನಿಜ;

    size_t memset_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, ದ್ವಿಗುಣ> shader_gpu_ಸಮಯ_ms;
    wgpu::ಬಫರ್_2                            profile_timestamp_dev_buf;
    wgpu::ಬಫರ್_2                            profile_timestamp_host_buf;
    wgpu::Query_ಗಣ                          profile_timestamp_query_ಗಣ;
    uint32_t                                profile_timestamp_query_ಎಣಿಕೆ = 0;
#ಆದರೆ_ಅಂತ್ಯ

    ~webgpu_context_struct() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
        ಆದರೆ (ಇದು->profile_timestamp_host_buf) {
            ಇದು->profile_timestamp_host_buf.Destroy();
            ಇದು->profile_timestamp_host_buf = ಶೂನ್ಯಸೂಚಕ;
        }
        ಆದರೆ (ಇದು->profile_timestamp_dev_buf) {
            ಇದು->profile_timestamp_dev_buf.Destroy();
            ಇದು->profile_timestamp_dev_buf = ಶೂನ್ಯಸೂಚಕ;
        }
        ಆದರೆ (ಇದು->profile_timestamp_query_ಗಣ) {
            ಇದು->profile_timestamp_query_ಗಣ.Destroy();
            ಇದು->profile_timestamp_query_ಗಣ = ಶೂನ್ಯಸೂಚಕ;
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (ಇದು->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf) {
            ಇದು->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf.Destroy();
            ಇದು->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf = ಶೂನ್ಯಸೂಚಕ;
        }
        ಆದರೆ (ಇದು->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf) {
            ಇದು->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf.Destroy();
            ಇದು->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf = ಶೂನ್ಯಸೂಚಕ;
        }
    }
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
ರಚನೆ ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_ಎಣಿಕೆ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *          name;
};

// Per-device struct for the global logical device interface
ರಚನೆ ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
ರಚನೆ ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
ರಚನೆ ggml_backend_webgpu_ಬಫರ್_context {
    wgpu::ಬಫರ್_2          ಬಫರ್;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_ಬಫರ್_context(wgpu::ಬಫರ್_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        ಬಫರ್(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

ಸ್ಥಿರ webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *                             shader_ಸಂಕೇತ,
                                                   ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *                             label,
                                                   ಸ್ಥಿರಾಂಕ std::vector<wgpu::Constant_ನಮೂದು> & constants = {}) {
    wgpu::Shader_ಮೂಲ_WGSL shader_ಮೂಲ;
    shader_ಮೂಲ.code = shader_ಸಂಕೇತ;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.ಮುಂದಿನ_ಒಳ_Chain = &shader_ಮೂಲ;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.ನಮೂದು_ಬಿಂದು_2 = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = ಶೂನ್ಯಸೂಚಕ;  // nullptr means auto layout
    ಆದರೆ (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.ದತ್ತಾಂಶ();
        pipeline_desc.compute.constant_ಎಣಿಕೆ = constants.size();
    }
    ಹಿಂತಿರುಗಿ { device.CreateComputePipeline(&pipeline_desc), label };
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_webgpu_create_ಬಫರ್(wgpu::Device &    device,
                                      wgpu::ಬಫರ್_2 &    ಬಫರ್,
                                      size_t            size,
                                      wgpu::ಬಫರ್_Usage usage,
                                      ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *      label) {
    wgpu::ಬಫರ್_Descriptor ಬಫರ್_desc;
    ಬಫರ್_desc.size             = size;
    ಬಫರ್_desc.usage            = usage;
    ಬಫರ್_desc.label            = label;
    ಬಫರ್_desc.mappedAtCreation = ಸುಳ್ಳು;

    // TODO: error handling
    ಬಫರ್ = device.Create_ಬಫರ್(&ಬಫರ್_desc);
}

ಸ್ಥಿರ wgpu::ಬಫರ್_2 ggml_webgpu_tensor_buf(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ggml_backend_webgpu_ಬಫರ್_context * ctx = (ggml_backend_webgpu_ಬಫರ್_context *) tensor->ಬಫರ್->context;
    ಹಿಂತಿರುಗಿ ctx->ಬಫರ್;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
ಸ್ಥಿರ size_t ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t alignment) {
    ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ    = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(t);
    ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(t->type);
    size_t       aligned   = ಸ್ಥಾನಾಂತರ & ~(alignment - 1);
    ಆಗ ((ಸ್ಥಾನಾಂತರ - aligned) % type_ಗಾತ್ರ != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    ಹಿಂತಿರುಗಿ aligned;
}

ಸ್ಥಿರ size_t ggml_webgpu_tensor_misalignment(ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t alignment) {
    ಹಿಂತಿರುಗಿ ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(t) - ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(t, alignment);
}

ಸ್ಥಿರ size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
}

ಸ್ಥಿರ size_t ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(webgpu_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(t, ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
}

ಸ್ಥಿರ size_t ggml_webgpu_tensor_binding_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ggml_tensor * t, size_t alignment) {
    ಹಿಂತಿರುಗಿ ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

ಸ್ಥಿರ size_t ggml_webgpu_tensor_binding_ಗಾತ್ರ(webgpu_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ ggml_webgpu_tensor_binding_ಗಾತ್ರ(t, ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_webgpu_tensor_binding_overlap(ಸ್ಥಿರಾಂಕ webgpu_global_context & global_ctx,
                                               ಸ್ಥಿರಾಂಕ ggml_tensor *           a,
                                               ಸ್ಥಿರಾಂಕ ggml_tensor *           b) {
    ಆದರೆ (a->ಬಫರ್ != b->ಬಫರ್) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ size_t alignment = global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;
    ಸ್ಥಿರಾಂಕ size_t a_ಸ್ಥಾನಾಂತರ  = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(a, alignment);
    ಸ್ಥಿರಾಂಕ size_t b_ಸ್ಥಾನಾಂತರ  = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(b, alignment);
    ಹಿಂತಿರುಗಿ a_ಸ್ಥಾನಾಂತರ < b_ಸ್ಥಾನಾಂತರ + ggml_webgpu_tensor_binding_ಗಾತ್ರ(b, alignment) &&
           b_ಸ್ಥಾನಾಂತರ < a_ಸ್ಥಾನಾಂತರ + ggml_webgpu_tensor_binding_ಗಾತ್ರ(a, alignment);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_webgpu_tensor_binding_overlap_range(ಸ್ಥಿರಾಂಕ webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_ಬಫರ್_t         ಬಫರ್,
                                                     size_t                        ಸ್ಥಾನಾಂತರ,
                                                     size_t                        size) {
    ಆದರೆ (tensor->ಬಫರ್ != ಬಫರ್) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ size_t alignment     = global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;
    ಸ್ಥಿರಾಂಕ size_t tensor_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(tensor, alignment);
    ಹಿಂತಿರುಗಿ tensor_ಸ್ಥಾನಾಂತರ < ಸ್ಥಾನಾಂತರ + size && ಸ್ಥಾನಾಂತರ < tensor_ಸ್ಥಾನಾಂತರ + ggml_webgpu_tensor_binding_ಗಾತ್ರ(tensor, alignment);
}

ರಚನೆ ggml_webgpu_merged_binding_range {
    size_t ಸ್ಥಾನಾಂತರ;
    size_t size;
};

ಸ್ಥಿರ ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_ಸ್ಥಾನಾಂತರ = SIZE_MAX;
    size_t merged_ಕೊನೆ    = 0;

    ಗಾಗಿ (ggml_tensor * tensor : tensors) {
        ಸ್ಥಿರಾಂಕ size_t bind_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, tensor);
        ಸ್ಥಿರಾಂಕ size_t bind_ಕೊನೆ    = bind_ಸ್ಥಾನಾಂತರ + ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, tensor);

        merged_ಸ್ಥಾನಾಂತರ = std::min(merged_ಸ್ಥಾನಾಂತರ, bind_ಸ್ಥಾನಾಂತರ);
        merged_ಕೊನೆ    = std::max(merged_ಕೊನೆ, bind_ಕೊನೆ);
    }

    ಹಿಂತಿರುಗಿ { merged_ಸ್ಥಾನಾಂತರ, merged_ಕೊನೆ - merged_ಸ್ಥಾನಾಂತರ };
}

ಸ್ಥಿರ uint32_t ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_tensor *                      tensor,
                                                         ಸ್ಥಿರಾಂಕ ggml_webgpu_merged_binding_range & merged_range) {
    ಹಿಂತಿರುಗಿ (uint32_t) ((ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(tensor) - merged_range.ಸ್ಥಾನಾಂತರ) / ggml_type_ಗಾತ್ರ(tensor->type));
}

ಸ್ಥಿರ wgpu::Bind_ಗುಂಪು_ನಮೂದು ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(uint32_t     binding,
                                                              wgpu::ಬಫರ್_2 ಬಫರ್,
                                                              uint64_t     ಸ್ಥಾನಾಂತರ,
                                                              uint64_t     size) {
    wgpu::Bind_ಗುಂಪು_ನಮೂದು ನಮೂದು = {};
    ನಮೂದು.binding              = binding;
    ನಮೂದು.ಬಫರ್               = std::move(ಬಫರ್);
    ನಮೂದು.ಸ್ಥಾನಾಂತರ               = ಸ್ಥಾನಾಂತರ;
    ನಮೂದು.size                 = size;
    ಹಿಂತಿರುಗಿ ನಮೂದು;
}

ಸ್ಥಿರ wgpu::Bind_ಗುಂಪು_ನಮೂದು ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    ಹಿಂತಿರುಗಿ ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, tensor),
                                             ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_check_wait_ಸ್ಥಿತಿ(wgpu::Wait_ಸ್ಥಿತಿ wait_ಸ್ಥಿತಿ,
                                                  T                callback_ಸ್ಥಿತಿ,
                                                  T                success_ಸ್ಥಿತಿ,
                                                  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *     wait_name,
                                                  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *     failure_name,
                                                  ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *     callback_ಸಂದೇಶ) {
    ಆದರೆ (wait_ಸ್ಥಿತಿ == wgpu::Wait_ಸ್ಥಿತಿ::Timed_ಹೊರಹೋಗು) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    ಆದರೆ (wait_ಸ್ಥಿತಿ == wgpu::Wait_ಸ್ಥಿತಿ::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    ಆದರೆ (callback_ಸ್ಥಿತಿ != success_ಸ್ಥಿತಿ) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<ಪೂರ್ಣಾಂಕ>(callback_ಸ್ಥಿತಿ),
                   callback_ಸಂದೇಶ);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
ಸ್ಥಿರ uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    ಹಿಂತಿರುಗಿ UINT32_MAX;
}

ಸ್ಥಿರ uint32_t ggml_backend_webgpu_get_command_submit_batch_ಗಾತ್ರ() {
    ಹಿಂತಿರುಗಿ WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_ಗಾತ್ರ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_wait_ಸರತಿ(webgpu_global_context & ctx) {
    wgpu::ಸರತಿ_Work_ಮುಗಿದ_ಸ್ಥಿತಿ callback_ಸ್ಥಿತಿ = wgpu::ಸರತಿ_Work_ಮುಗಿದ_ಸ್ಥಿತಿ::Error;
    std::string               callback_ಸಂದೇಶ;

    ಸ್ಥಿರಾಂಕ wgpu::Wait_ಸ್ಥಿತಿ wait_ಸ್ಥಿತಿ = ctx->instance.WaitAny(
        ctx->queue.On_Submitted_Work_ಮುಗಿದ(
            wgpu::Callback_ಬಹುಲಕ::AllowSpontaneous,
            [&callback_ಸ್ಥಿತಿ, &callback_ಸಂದೇಶ](wgpu::ಸರತಿ_Work_ಮುಗಿದ_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2, wgpu::StringView ಸಂದೇಶ) {
                callback_ಸ್ಥಿತಿ  = ಸ್ಥಿತಿ_2;
                callback_ಸಂದೇಶ = std::string(ಸಂದೇಶ);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_ಸ್ಥಿತಿ(wait_ಸ್ಥಿತಿ, callback_ಸ್ಥಿತಿ, wgpu::ಸರತಿ_Work_ಮುಗಿದ_ಸ್ಥಿತಿ::Success,
                                          "Queue wait", "Queue work", callback_ಸಂದೇಶ.c_str());
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_map_ಬಫರ್(webgpu_global_context & ctx,
                                           wgpu::ಬಫರ್_2 &          ಬಫರ್,
                                           wgpu::Map_ಬಹುಲಕ           mode,
                                           size_t                  ಸ್ಥಾನಾಂತರ,
                                           size_t                  size) {
    wgpu::Map_Async_ಸ್ಥಿತಿ callback_ಸ್ಥಿತಿ = wgpu::Map_Async_ಸ್ಥಿತಿ::Error;
    std::string          callback_ಸಂದೇಶ;

    ಸ್ಥಿರಾಂಕ wgpu::Wait_ಸ್ಥಿತಿ wait_ಸ್ಥಿತಿ = ctx->instance.WaitAny(
        ಬಫರ್.MapAsync(mode, ಸ್ಥಾನಾಂತರ, size, wgpu::Callback_ಬಹುಲಕ::AllowSpontaneous,
                        [&callback_ಸ್ಥಿತಿ, &callback_ಸಂದೇಶ](wgpu::Map_Async_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2, wgpu::StringView ಸಂದೇಶ) {
                            callback_ಸ್ಥಿತಿ  = ಸ್ಥಿತಿ_2;
                            callback_ಸಂದೇಶ = std::string(ಸಂದೇಶ);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_ಸ್ಥಿತಿ(wait_ಸ್ಥಿತಿ, callback_ಸ್ಥಿತಿ, wgpu::Map_Async_ಸ್ಥಿತಿ::Success,
                                          "Buffer map wait", "Buffer map", callback_ಸಂದೇಶ.c_str());
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ಸ್ಥಿರಾಂಕ wgpu::Command_ಬಫರ್ commands,
                                                uint32_t &                ಅಂಕೆ_inflight_batches) {
    ಆದರೆ (ಅಂಕೆ_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_ಸರತಿ(ctx->global_ctx);
        ಅಂಕೆ_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    ಅಂಕೆ_inflight_batches++;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_ಬಫರ್_ಗಮ್ಯ_ಬಫರ್(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_ಗಾತ್ರ());
    wgpu::Command_ಬಫರ್ commands = encoder.ಮುಕ್ತಾಯ();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_ಬಫರ್(ctx, ctx->debug_host_buf, wgpu::Map_ಬಹುಲಕ::Read, 0, ctx->debug_host_buf.Get_ಗಾತ್ರ());
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * debug_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ctx->debug_host_buf.GetConstMappedRange();
    std::ಹೊರಗೆ << "debug[0]: " << debug_ದತ್ತಾಂಶ[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ಸ್ಥಿರಾಂಕ std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            ಫಲಿತಾಂಶ = {};
    std::vector<wgpu::Bind_ಗುಂಪು> bind_groups;
    std::vector<size_t>          param_offsets;
    ಫಲಿತಾಂಶ.ಅಂಕೆ_kernels = dispatches.size();

    ಗಾಗಿ (size_t i = 0; i < dispatches.size(); i++) {
        ಸ್ಥಿರಾಂಕ webgpu_dispatch_desc & dispatch     = dispatches[i];
        ಸ್ಥಿರಾಂಕ size_t                 param_ಗಾತ್ರ   = dispatch.params.size() * ಗಾತ್ರ(uint32_t);
        ಸ್ಥಿರಾಂಕ size_t                 param_ಸ್ಥಾನಾಂತರ = ctx->param_arena.alloc_ಸ್ಥಾನ(param_ಗಾತ್ರ);

        std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries            = dispatch.bind_ಗುಂಪು_entries;
        uint32_t                          params_binding_ಅಂಕೆ = entries.size();
        entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(params_binding_ಅಂಕೆ, ctx->param_arena.ಬಫರ್, param_ಸ್ಥಾನಾಂತರ,
                                                            ctx->param_arena.ಸ್ಥಾನ_ಗಾತ್ರ));

        wgpu::Bind_ಗುಂಪು_Descriptor bind_ಗುಂಪು_desc;
        bind_ಗುಂಪು_desc.layout     = dispatch.pipeline.pipeline.Get_Bind_ಗುಂಪು_Layout(0);
        bind_ಗುಂಪು_desc.ನಮೂದು_ಎಣಿಕೆ = entries.size();
        bind_ಗುಂಪು_desc.entries    = entries.ದತ್ತಾಂಶ();
        bind_ಗುಂಪು_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.Create_Bind_ಗುಂಪು(&bind_ಗುಂಪು_desc));
        param_offsets.push_back(param_ಸ್ಥಾನಾಂತರ);
    }

    ಗಾಗಿ (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_ಬಫರ್(ctx->param_arena.ಬಫರ್, param_offsets[i], dispatches[i].params.ದತ್ತಾಂಶ(),
                                           dispatches[i].params.size() * ಗಾತ್ರ(uint32_t));
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    ಗಾಗಿ (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_ಎಣಿಕೆ + 2 <= WEBGPU_MAX_PROFILE_QUERY_ಎಣಿಕೆ);
        ಸ್ಥಿರಾಂಕ uint32_t query_begin = ctx->profile_timestamp_query_ಎಣಿಕೆ++;
        ಸ್ಥಿರಾಂಕ uint32_t query_ಕೊನೆ   = ctx->profile_timestamp_query_ಎಣಿಕೆ++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.query_ಗಣ                    = ctx->profile_timestamp_query_ಗಣ;
        ts_writes.beginning_ದ_Pass_Write_ಸೂಚ್ಯಂಕ   = query_begin;
        ts_writes.ಕೊನೆ_ದ_Pass_Write_ಸೂಚ್ಯಂಕ         = query_ಕೊನೆ;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->ಸಕ್ರಿಯ_command_encoder.BeginComputePass(&pass_desc);

        pass.ಗಣ_Pipeline(dispatches[i].pipeline.pipeline);
        pass.ಗಣ_Bind_ಗುಂಪು(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.ಮೊದಲ, dispatches[i].workgroups.second, 1);
        pass.ಕೊನೆ_2();
        ಫಲಿತಾಂಶ.pipeline_ಹೆಸರುಗಳು.push_back(dispatches[i].pipeline.name);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಗಾಗಿ (size_t i = 0; i < dispatches.size(); i++) {
        ಆದರೆ (ctx->batch_compute_passes) {
            ctx->ಸಕ್ರಿಯ_compute_pass.ಗಣ_Pipeline(dispatches[i].pipeline.pipeline);
            ctx->ಸಕ್ರಿಯ_compute_pass.ಗಣ_Bind_ಗುಂಪು(0, bind_groups[i]);
            ctx->ಸಕ್ರಿಯ_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.ಮೊದಲ, dispatches[i].workgroups.second,
                                                        1);
        } ಇಲ್ಲದಿದ್ದರೆ {
            wgpu::ComputePassEncoder pass = ctx->ಸಕ್ರಿಯ_command_encoder.BeginComputePass();
            pass.ಗಣ_Pipeline(dispatches[i].pipeline.pipeline);
            pass.ಗಣ_Bind_ಗುಂಪು(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.ಮೊದಲ, dispatches[i].workgroups.second, 1);
            pass.ಕೊನೆ_2();
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> bind_ಗುಂಪು_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_ಗುಂಪು_entries), { wg_x, wg_y } },
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_ಬಫರ್_memset(webgpu_global_context & ctx,
                                              wgpu::ಬಫರ್_2 &          buf,
                                              uint32_t                value,
                                              size_t                  ಸ್ಥಾನಾಂತರ,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) ಸ್ಥಾನಾಂತರ, (uint32_t) size, value };
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, buf, 0, buf.Get_ಗಾತ್ರ()) };
    size_t                            ಬೈಟುಗಳು_ಪ್ರತಿ_wg =
        ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup * ctx->capabilities.memset_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, ಬೈಟುಗಳು_ಪ್ರತಿ_wg);

    ctx->queue.Write_ಬಫರ್(ctx->memset_params_buf, 0, params.ದತ್ತಾಂಶ(), params.size() * ಗಾತ್ರ(uint32_t));

    wgpu::Bind_ಗುಂಪು_ನಮೂದು params_ನಮೂದು = {};
    params_ನಮೂದು.binding              = 1;
    params_ನಮೂದು.ಬಫರ್               = ctx->memset_params_buf;
    params_ನಮೂದು.ಸ್ಥಾನಾಂತರ               = 0;
    params_ನಮೂದು.size                 = WEBGPU_PARAMS_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು;
    entries.push_back(params_ನಮೂದು);

    wgpu::Bind_ಗುಂಪು_Descriptor bind_ಗುಂಪು_desc;
    bind_ಗುಂಪು_desc.layout     = ctx->memset_pipeline.pipeline.Get_Bind_ಗುಂಪು_Layout(0);
    bind_ಗುಂಪು_desc.ನಮೂದು_ಎಣಿಕೆ = entries.size();
    bind_ಗುಂಪು_desc.entries    = entries.ದತ್ತಾಂಶ();
    bind_ಗುಂಪು_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::Bind_ಗುಂಪು bind_ಗುಂಪು = ctx->device.Create_Bind_ಗುಂಪು(&bind_ಗುಂಪು_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.ಗಣ_Pipeline(ctx->memset_pipeline.pipeline);
    pass.ಗಣ_Bind_ಗುಂಪು(0, bind_ಗುಂಪು);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.ಕೊನೆ_2();

    wgpu::Command_ಬಫರ್              command  = encoder.ಮುಕ್ತಾಯ();
    std::vector<wgpu::Command_ಬಫರ್> commands = { command };
    ctx->queue.Submit(commands.size(), commands.ದತ್ತಾಂಶ());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_CPU_PROFILE
    std::ಹೊರಗೆ << "\n[ggml_webgpu cpu profiling summary]\n";
    ದ್ವಿಗುಣ ಒಟ್ಟು_cpu = 0.0;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kv : ctx->webgpu_ctx->global_ctx->cpu_ಸಮಯ_ms) {
        ಒಟ್ಟು_cpu += kv.second;
    }
    std::ಹೊರಗೆ << "ggml_webgpu: total cpu time: " << ಒಟ್ಟು_cpu << " ms\n";
    std::ಹೊರಗೆ << "ggml_webgpu: cpu breakdown:\n";
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kv : ctx->webgpu_ctx->global_ctx->cpu_ಸಮಯ_ms) {
        ದ್ವಿಗುಣ pct = (ಒಟ್ಟು_cpu > 0.0) ? (kv.second / ಒಟ್ಟು_cpu * 100.0) : 0.0;
        std::ಹೊರಗೆ << "ggml_webgpu:  " << kv.ಮೊದಲ << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    ಆದರೆ (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::ಹೊರಗೆ << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        ದ್ವಿಗುಣ pct = (ಒಟ್ಟು_cpu > 0.0) ? (kv.second / ಒಟ್ಟು_cpu * 100.0) : 0.0;
        std::ಹೊರಗೆ << "ggml_webgpu:  " << kv.ಮೊದಲ << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    std::ಹೊರಗೆ << "\n[ggml_webgpu gpu profiling summary]\n";
    ದ್ವಿಗುಣ ಒಟ್ಟು_gpu = 0.0;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kv : ctx->webgpu_ctx->shader_gpu_ಸಮಯ_ms) {
        ಒಟ್ಟು_gpu += kv.second;
    }
    std::ಹೊರಗೆ << "ggml_webgpu: total gpu time (all shaders): " << ಒಟ್ಟು_gpu << " ms\n";
    std::ಹೊರಗೆ << "\nggml_webgpu: gpu breakdown:\n";
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kv : ctx->webgpu_ctx->shader_gpu_ಸಮಯ_ms) {
        ದ್ವಿಗುಣ pct = (ಒಟ್ಟು_gpu > 0.0) ? (kv.second / ಒಟ್ಟು_gpu * 100.0) : 0.0;
        std::ಹೊರಗೆ << "ggml_webgpu:  " << kv.ಮೊದಲ << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::ಹೊರಗೆ << "ggml_webgpu: gpu/cpu ratio: " << (ಒಟ್ಟು_cpu > 0.0 ? ಒಟ್ಟು_gpu / ಒಟ್ಟು_cpu : 0.0) << "\n";
#ಆದರೆ_ಅಂತ್ಯ

    ಅಳಿಸು ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_ಗಾತ್ರ(src->type)), (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)), (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_ಗಾತ್ರ(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ne, decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_ಗಣ(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ಗಣ_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ inplace   = decisions->inplace;

    ಸ್ಥಿರಾಂಕ uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = (uint32_t) ggml_type_ಗಾತ್ರ(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[3] / dst_type_ಗಾತ್ರ),

        (uint32_t) (src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),

        1u,
        (uint32_t) (((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[0] / dst_type_ಗಾತ್ರ),
        (uint32_t) (((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[1] / dst_type_ಗಾತ್ರ),
        (uint32_t) (((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[2] / dst_type_ಗಾತ್ರ),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries;
    uint32_t                          binding_ಸೂಚ್ಯಂಕ = 0;
    ಆದರೆ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0));
        binding_ಸೂಚ್ಯಂಕ++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಸೂಚ್ಯಂಕ, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಸೂಚ್ಯಂಕ + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ne, decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ಸ್ಥಿರಾಂಕ uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)),
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

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ne, decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
    shader_lib_ctx.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_ಗಾತ್ರ;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst),
    };

    ಸ್ಥಿರಾಂಕ uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    ಸ್ಥಿರಾಂಕ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ಸ್ಥಿರಾಂಕ int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ಸ್ಥಿರಾಂಕ int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ಸ್ಥಿರಾಂಕ int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),

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

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    ಸ್ಥಿರಾಂಕ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ಸ್ಥಿರಾಂಕ int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ಸ್ಥಿರಾಂಕ int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ಸ್ಥಿರಾಂಕ int32_t d1 = ggml_get_op_params_i32(dst, 5);

    // Scalar params matching conv2d_dw.wgsl (weight src0 [KW,KH,1,C], input src1, output dst).
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),

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

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    ಸ್ಥಿರಾಂಕ int32_t s0    = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t s1    = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ int32_t p0    = ggml_get_op_params_i32(dst, 2);
    ಸ್ಥಿರಾಂಕ int32_t p1    = ggml_get_op_params_i32(dst, 3);
    ಸ್ಥಿರಾಂಕ int32_t d0    = ggml_get_op_params_i32(dst, 4);
    ಸ್ಥಿರಾಂಕ int32_t d1    = ggml_get_op_params_i32(dst, 5);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    ಸ್ಥಿರಾಂಕ uint32_t KW = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t KH = is_2D ? src0->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    ಸ್ಥಿರಾಂಕ uint32_t IW = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t IH = is_2D ? src1->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    ಸ್ಥಿರಾಂಕ uint32_t OW = dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t OH = is_2D ? dst->ne[2] : 1;

    ಸ್ಥಿರಾಂಕ uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type));
    ಸ್ಥಿರಾಂಕ uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)) : 0;
    ಸ್ಥಿರಾಂಕ uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type));
    ಸ್ಥಿರಾಂಕ uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type));

    ಸ್ಥಿರಾಂಕ uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_ಗಾತ್ರ(dst->type));
    ಸ್ಥಿರಾಂಕ uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type));
    ಸ್ಥಿರಾಂಕ uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)) : 0;
    ಸ್ಥಿರಾಂಕ uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),

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

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ಸ್ಥಿರಾಂಕ uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->ಟೋಕನುಗಳು_ಪ್ರತಿ_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst),
    };

    ಸ್ಥಿರಾಂಕ uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->ಖಂಡ_ಗಾತ್ರ_3);
    ಸ್ಥಿರಾಂಕ uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_ssm_ಪರಿಶೀಲನೆ(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_ಗಾತ್ರ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    ಆದರೆ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    ಆದರೆ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    ಆದರೆ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_ಪರಿಶೀಲನೆ_pipeline(shader_lib_ctx, xbc_overlap);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_ssm_ಪರಿಶೀಲನೆ_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t ಸ್ಥಾನಾಂತರ_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type));
    uint32_t ಸ್ಥಾನಾಂತರ_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type));
    uint32_t ಸ್ಥಾನಾಂತರ_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_ಗಾತ್ರ(src3->type));
    uint32_t ಸ್ಥಾನಾಂತರ_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_ಗಾತ್ರ(src4->type));
    uint32_t ಸ್ಥಾನಾಂತರ_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_ಗಾತ್ರ(src5->type));
    uint32_t ಸ್ಥಾನಾಂತರ_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_ಗಾತ್ರ(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    ಆದರೆ (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        ಸ್ಥಾನಾಂತರ_x         = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src1, xbc_merged_range);
        ಸ್ಥಾನಾಂತರ_B         = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src4, xbc_merged_range);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        ಸ್ಥಾನಾಂತರ_B         = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src4, xbc_merged_range);
        ಸ್ಥಾನಾಂತರ_C         = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src5, xbc_merged_range);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        ಸ್ಥಾನಾಂತರ_x         = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src1, xbc_merged_range);
        ಸ್ಥಾನಾಂತರ_B         = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src4, xbc_merged_range);
        ಸ್ಥಾನಾಂತರ_C         = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_ಸ್ಥಾನಾಂತರ = 0;
    size_t xbc_bind_ಗಾತ್ರ   = 0;
    ಆದರೆ (xbc_overlap > 0) {
        xbc_bind_ಸ್ಥಾನಾಂತರ = xbc_merged_range.ಸ್ಥಾನಾಂತರ;
        xbc_bind_ಗಾತ್ರ   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        ಸ್ಥಾನಾಂತರ_x,
        ಸ್ಥಾನಾಂತರ_dt,
        ಸ್ಥಾನಾಂತರ_A,
        ಸ್ಥಾನಾಂತರ_B,
        ಸ್ಥಾನಾಂತರ_C,
        ಸ್ಥಾನಾಂತರ_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_ಗಾತ್ರ(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_ಗಾತ್ರ(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_ಗಾತ್ರ(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_ಗಾತ್ರ(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_ಗಾತ್ರ(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_ಗಾತ್ರ(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_ಗಾತ್ರ(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_ಗಾತ್ರ(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_ಗಾತ್ರ(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_ಅಂಕೆ = 0;

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, src0),
    };
    // xbc_merged binding
    ಆದರೆ (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(binding_ಅಂಕೆ++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_ಸ್ಥಾನಾಂತರ, xbc_bind_ಗಾತ್ರ));
    }
    // x
    ಆದರೆ (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, src1));
    }
    // B
    ಆದರೆ (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, src4));
    }
    // C
    ಆದರೆ (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ++, dst));

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ಸ್ಥಿರಾಂಕ uint32_t max_wg_ಪ್ರತಿ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(ಒಟ್ಟು_wg, max_wg_ಪ್ರತಿ_dim, wg_x, wg_y);

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ಸ್ಥಿರಾಂಕ uint32_t s_v      = (uint32_t) src2->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t h        = (uint32_t) src2->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = (uint32_t) src2->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    scale    = 1.0f / sqrtf((ದಶಮಾಂಶ) s_v);
    uint32_t       scale_u32;
    ಸ್ಮರಣೆ_ನಕಲು(&scale_u32, &scale, ಗಾತ್ರ(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_ಟೋಕನುಗಳು,
        n_seqs,
        s_v * h * n_ಟೋಕನುಗಳು * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_ಗಾತ್ರ(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_ಗಾತ್ರ(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_ಗಾತ್ರ(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_ಗಾತ್ರ(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_ಗಾತ್ರ(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_ಗಾತ್ರ(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_ಗಾತ್ರ(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_ಗಾತ್ರ(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0), ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, src2), ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 4, src4), ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 6, dst),
    };

    ಆದರೆ (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 7, dst_fuse));
    }

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

ಸ್ಥಿರ std::optional<webgpu_encoded_op> ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(src) || ggml_is_ಖಾಲಿಗಳು(idx)) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_ಗಾತ್ರ(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)), (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)), (uint32_t) (idx->nb[0] / ggml_type_ಗಾತ್ರ(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_ಗಾತ್ರ(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_ಗಾತ್ರ(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst));

    ಆದರೆ (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(3, ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf, 0,
                                                            ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf.Get_ಗಾತ್ರ()));
    }

    uint32_t ಎಳೆಗಳು;
    ಆದರೆ (ggml_is_quantized(dst->type)) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = src->ne[0] / ggml_blck_ಗಾತ್ರ(dst->type);
        ಎಳೆಗಳು =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_ಖಂಡಗಳು ? (ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು / 2) : ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (decisions->vec4) {
        ಎಳೆಗಳು = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಎಳೆಗಳು = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ಎಳೆಗಳು, decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
ಸ್ಥಿರ std::vector<wgpu::Constant_ನಮೂದು> ggml_webgpu_wg_ಗಾತ್ರ_ನಮೂದು(uint32_t wg_ಗಾತ್ರ) {
    std::vector<wgpu::Constant_ನಮೂದು> constants(1);
    constants[0].ಕೀಲಿ   = "wg_size";
    constants[0].value = wg_ಗಾತ್ರ;
    ಹಿಂತಿರುಗಿ constants;
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ಶೂನ್ಯಸೂಚಕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type));
    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type));
    ಸ್ಥಿರಾಂಕ uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type));
    ಸ್ಥಿರಾಂಕ uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type));
    ಸ್ಥಿರಾಂಕ uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type));

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ vec4_aligned = ಸ್ಥಾನಾಂತರ_src % 4 == 0 && ಸ್ಥಾನಾಂತರ_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_ಅಡ್ಡಸಾಲುಗಳು_pipeline(shader_lib_ctx, vec4_aligned);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { ಸ್ಥಾನಾಂತರ_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_ಗಾತ್ರ(idx->type)),
                                     ಸ್ಥಾನಾಂತರ_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_ಗಾತ್ರ(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_ಗಾತ್ರ(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_ಗಾತ್ರ(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst) };

    uint32_t ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t ಒಟ್ಟು_ಎಳೆಗಳು  = float_parallel ? ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು * ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು : ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ಒಟ್ಟು_ಎಳೆಗಳು, decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ           = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(dst);
    ಸ್ಥಿರಾಂಕ size_t q8_src1_align_ಸ್ಥಾನಾಂತರ = ROUNDUP_POW2(
        dst_ಸ್ಥಾನಾಂತರ + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
    ಸ್ಥಿರಾಂಕ size_t q8_src1_binding_ಗಾತ್ರ = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> q8_entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src1),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_ಸ್ಥಾನಾಂತರ, q8_src1_binding_ಗಾತ್ರ)
    };

    ಸ್ವಯಂಚಾಲಿತ q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_ಗಾತ್ರ     = q8_decisions->wg_ಗಾತ್ರ;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ಸ್ಥಿರಾಂಕ uint32_t wg_ಪ್ರತಿ_vec     = (src0->ne[0] / 4 + (q8_wg_ಗಾತ್ರ - 1)) / q8_wg_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t q8_ಒಟ್ಟು_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_ಪ್ರತಿ_vec;
    ಸ್ಥಿರಾಂಕ uint32_t max_wg_ಪ್ರತಿ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension;
    compute_2d_workgroups(q8_ಒಟ್ಟು_wg, max_wg_ಪ್ರತಿ_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    ತಾರ್ಕಿಕ use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    ತಾರ್ಕಿಕ use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_ಗುಣಲಬ್ಧ,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_ಮಾತೃಕೆ = ctx->global_ctx->capabilities.supports_subgroup_ಮಾತೃಕೆ;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_ಗಾತ್ರ        = ctx->global_ctx->capabilities.min_subgroup_ಗಾತ್ರ;
    shader_lib_ctx.max_subgroup_ಗಾತ್ರ        = ctx->global_ctx->capabilities.max_subgroup_ಗಾತ್ರ;
    shader_lib_ctx.supports_dot_ಗುಣಲಬ್ಧ     = ctx->global_ctx->capabilities.supports_dot_ಗುಣಲಬ್ಧ;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    ಆದರೆ (use_mat_vec) {
        ಆದರೆ (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } ಇಲ್ಲದಿದ್ದರೆ {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t ಸ್ಥಾನಾಂತರ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type));
    uint32_t ಸ್ಥಾನಾಂತರ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type));
    size_t   merged_ಸ್ಥಾನಾಂತರ = 0;
    size_t   merged_ಗಾತ್ರ   = 0;
    ಆದರೆ (src_overlap) {
        ಸ್ಥಿರಾಂಕ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_ಸ್ಥಾನಾಂತರ = merged_range.ಸ್ಥಾನಾಂತರ;
        merged_ಗಾತ್ರ   = merged_range.size;
        ಸ್ಥಾನಾಂತರ_src0   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src0, merged_range);
        ಸ್ಥಾನಾಂತರ_src1   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { ಸ್ಥಾನಾಂತರ_src0,
                                     ಸ್ಥಾನಾಂತರ_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {};
    ಆದರೆ (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0));
        ಸ್ವಯಂಚಾಲಿತ & mmvq_qq8_ನಮೂದು = dispatches[0].bind_ಗುಂಪು_entries[1];
        entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_ನಮೂದು.ಸ್ಥಾನಾಂತರ,
                                                            mmvq_qq8_ನಮೂದು.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src0), merged_ಸ್ಥಾನಾಂತರ, merged_ಗಾತ್ರ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst));
    } ಇಲ್ಲದಿದ್ದರೆ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ಸ್ಥಿರಾಂಕ uint32_t max_wg_ಪ್ರತಿ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension;

    ಆದರೆ (use_mat_vec) {
        ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_ಪ್ರತಿ_wg);
        uint32_t ಒಟ್ಟು_wg      = output_groups * batches;
        compute_2d_workgroups(ಒಟ್ಟು_wg, max_wg_ಪ್ರತಿ_dim, wg_x, wg_y);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        ಆದರೆ (decisions->use_subgroup_ಮಾತೃಕೆ) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_ಮಾತೃಕೆ_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_ಮಾತೃಕೆ_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } ಇಲ್ಲದಿದ್ದರೆ {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_ಗಾತ್ರ_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_ಗಾತ್ರ_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t ಒಟ್ಟು_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(ಒಟ್ಟು_wg, max_wg_ಪ್ರತಿ_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    ಸ್ಥಿರಾಂಕ uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t param_n_expert_ಬಳಸಿದ = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_ಬಳಸಿದ,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src0),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src0)),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src1),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src1)),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src2),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src2)),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, dst),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ಸ್ಥಿರಾಂಕ uint32_t max_wg_ಪ್ರತಿ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_ಪ್ರತಿ_wg);
    uint32_t       ಒಟ್ಟು_wg       = output_groups * param_n_expert_ಬಳಸಿದ;
    compute_2d_workgroups(ಒಟ್ಟು_wg, max_wg_ಪ್ರತಿ_dim, wg_x, wg_y);

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
ಸ್ಥಿರ uint32_t ggml_webgpu_mul_mat_id_max_tiles(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    ಹಿಂತಿರುಗಿ CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
ಸ್ಥಿರ size_t ggml_webgpu_mul_mat_id_tiles_nbytes(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * ಗಾತ್ರ(uint32_t);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    ಆದರೆ (dst->ne[2] == 1) {
        ಹಿಂತಿರುಗಿ ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ಸ್ಥಿರಾಂಕ uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t param_n_expert_ಬಳಸಿದ = (uint32_t) dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t param_n_ಟೋಕನುಗಳು      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type)),
        param_n_expert,
        param_n_expert_ಬಳಸಿದ,
        param_n_ಟೋಕನುಗಳು,
        (uint32_t) (src2->nb[1] / ggml_type_ಗಾತ್ರ(src2->type)),
    };

    ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ       = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(dst);
    ಸ್ಥಿರಾಂಕ size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * ಗಾತ್ರ(uint32_t);
    ಸ್ಥಿರಾಂಕ size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ಸ್ಥಿರಾಂಕ size_t ids_cache_align_ಸ್ಥಾನಾಂತರ = ROUNDUP_POW2(
        dst_ಸ್ಥಾನಾಂತರ + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
    ಸ್ಥಿರಾಂಕ size_t ids_tiles_align_ಸ್ಥಾನಾಂತರ =
        ROUNDUP_POW2(ids_cache_align_ಸ್ಥಾನಾಂತರ + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);

    ಸ್ಥಿರಾಂಕ size_t ids_cache_bind_ಗಾತ್ರ = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ಸ್ಥಿರಾಂಕ size_t ids_tiles_bind_ಗಾತ್ರ = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> gather_entries = {
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src2),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src2)),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_ಸ್ಥಾನಾಂತರ, ids_cache_bind_ಗಾತ್ರ),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_ಸ್ಥಾನಾಂತರ, ids_tiles_bind_ಗಾತ್ರ),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_ಬಳಸಿದ,
        param_n_ಟೋಕನುಗಳು,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> main_entries = {
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src0),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src0)),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src1),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src1)),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, dst),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, dst)),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_ಸ್ಥಾನಾಂತರ, ids_cache_bind_ಗಾತ್ರ),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_ಸ್ಥಾನಾಂತರ, ids_tiles_bind_ಗಾತ್ರ),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    ಸ್ವಯಂಚಾಲಿತ * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_ಗಾತ್ರ_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_ಗಾತ್ರ_N);

    ಸ್ಥಿರಾಂಕ uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_ಗಾತ್ರ_m;
    ಸ್ಥಿರಾಂಕ uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ರಚನೆ ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries;
    size_t                            kv_bind_ಸ್ಥಾನಾಂತರ = 0;
    size_t                            kv_bind_ಗಾತ್ರ   = 0;
    ತಾರ್ಕಿಕ                              has_ಮುಖವಾಡ       = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ                              has_sinks      = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ                              kv_overlap     = ಸುಳ್ಳು;
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_webgpu_flash_attn_use_vec_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ webgpu_global_context & global_ctx,
                                                ಸ್ಥಿರಾಂಕ ggml_tensor *           Q,
                                                ಸ್ಥಿರಾಂಕ ggml_tensor *           K,
                                                ಸ್ಥಿರಾಂಕ ggml_tensor *           V) {
    ಸ್ಥಿರಾಂಕ size_t storage_ಸ್ಥಾನಾಂತರ_alignment = global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_ಸ್ಥಾನಾಂತರ_alignment);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_ಸ್ಥಾನಾಂತರ_alignment);

    ಸ್ಥಿರಾಂಕ uint32_t k_vec_ತಲೆ_align =
        ggml_is_quantized(K->type) ? ggml_blck_ಗಾತ್ರ(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ;
    ಸ್ಥಿರಾಂಕ uint32_t v_vec_ತಲೆ_align =
        ggml_is_quantized(V->type) ? ggml_blck_ಗಾತ್ರ(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_vec_ತಲೆ_dims_aligned = Q->ne[0] % k_vec_ತಲೆ_align == 0 && V->ne[0] % v_vec_ತಲೆ_align == 0;

    ಹಿಂತಿರುಗಿ global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_ತಲೆ_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

ಸ್ಥಿರ ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    ಮುಖವಾಡ,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    ದಶಮಾಂಶ scale         = ggml_get_op_params_f32(dst, 0);
    ದಶಮಾಂಶ max_bias      = ggml_get_op_params_f32(dst, 1);
    ದಶಮಾಂಶ logit_softcap = ggml_get_op_params_f32(dst, 2);
    ಆದರೆ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    ದಶಮಾಂಶ n_ತಲೆ_log2 = ದಶಮಾಂಶ(1u << (uint32_t) floor(log2(Q->ne[2])));
    ದಶಮಾಂಶ m0          = powf(2.0f, -(max_bias) / n_ತಲೆ_log2);
    ದಶಮಾಂಶ m1          = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = ಮುಖವಾಡ;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_ಮಾತೃಕೆ = ctx->global_ctx->capabilities.supports_subgroup_ಮಾತೃಕೆ;
    op.shader_lib_ctx.max_wg_ಗಾತ್ರ              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
    op.shader_lib_ctx.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_ಗಾತ್ರ;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_ಗಾತ್ರ        = ctx->global_ctx->capabilities.min_subgroup_ಗಾತ್ರ;
    op.shader_lib_ctx.max_subgroup_ಗಾತ್ರ        = ctx->global_ctx->capabilities.max_subgroup_ಗಾತ್ರ;

    op.has_ಮುಖವಾಡ   = ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ;
    op.has_sinks  = sinks != ಶೂನ್ಯಸೂಚಕ;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t ಸ್ಥಾನಾಂತರ_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_ಗಾತ್ರ(K->type));
    uint32_t ಸ್ಥಾನಾಂತರ_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_ಗಾತ್ರ(V->type));
    ಆದರೆ (op.kv_overlap) {
        ಸ್ಥಿರಾಂಕ ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_ಸ್ಥಾನಾಂತರ                                   = merged_range.ಸ್ಥಾನಾಂತರ;
        op.kv_bind_ಗಾತ್ರ                                     = merged_range.size;
        ಸ್ಥಾನಾಂತರ_k                                            = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(K, merged_range);
        ಸ್ಥಾನಾಂತರ_v                                            = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_ಗಾತ್ರ(Q->type)),
        ಸ್ಥಾನಾಂತರ_k,
        ಸ್ಥಾನಾಂತರ_v,
        op.has_ಮುಖವಾಡ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ಮುಖವಾಡ) / ggml_type_ಗಾತ್ರ(ಮುಖವಾಡ->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_ಗಾತ್ರ(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_ಗಾತ್ರ(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_ಗಾತ್ರ(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_ಗಾತ್ರ(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_ಗಾತ್ರ(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_ಗಾತ್ರ(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_ಗಾತ್ರ(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_ಗಾತ್ರ(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_ಗಾತ್ರ(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_ಗಾತ್ರ(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_ಮುಖವಾಡ ? (uint32_t) (ಮುಖವಾಡ->nb[3] / ggml_type_ಗಾತ್ರ(ಮುಖವಾಡ->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_ತಲೆ_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, Q),
    };
    ಆದರೆ (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(K), op.kv_bind_ಸ್ಥಾನಾಂತರ, op.kv_bind_ಗಾತ್ರ));
    } ಇಲ್ಲದಿದ್ದರೆ {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, V));
    }
    uint32_t binding_ಸೂಚ್ಯಂಕ = op.kv_overlap ? 2u : 3u;
    ಆದರೆ (op.has_ಮುಖವಾಡ) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಸೂಚ್ಯಂಕ++, ಮುಖವಾಡ));
    }
    ಆದರೆ (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಸೂಚ್ಯಂಕ++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಸೂಚ್ಯಂಕ++, dst));

    ಹಿಂತಿರುಗಿ op;
}

ಸ್ಥಿರ uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ಸ್ಥಿರಾಂಕ uint64_t kv_ಹರವು = (uint64_t) kv_tile;
    ಆಗ ((2u * nwg * kv_ಹರವು) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    ಹಿಂತಿರುಗಿ std::min(nwg, vec_nwg_cap);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_flash_attn_ನೇರ(webgpu_context & ctx, ಸ್ಥಿರಾಂಕ ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    ಸ್ವಯಂಚಾಲಿತ *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_ಪ್ರತಿ_ತಲೆ = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_ಪ್ರತಿ_ತಲೆ * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             ಮುಖವಾಡ,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::ಬಫರ್_2 blk_buf         = {};
    uint64_t     blk_ಗಾತ್ರ_ಬೈಟುಗಳು  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_ಎಣಿಕೆ = 0;

    ಸ್ಥಿರಾಂಕ uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_ಗಾತ್ರ;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ಸ್ಥಿರಾಂಕ uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_ತಳ  = 0;
    uint64_t     tmp_ಗಾತ್ರ_ಬೈಟುಗಳು  = 0;
    wgpu::ಬಫರ್_2 tmp_buf         = {};
    uint64_t     tmp_bind_ಸ್ಥಾನಾಂತರ = 0;
    uint64_t     tmp_bind_ಗಾತ್ರ   = 0;
    ಸ್ಥಿರಾಂಕ size_t align_ಬೈಟುಗಳು     = ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;
    ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ      = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(dst);
    size_t       scratch_ಸ್ಥಾನಾಂತರ  = ROUNDUP_POW2(dst_ಸ್ಥಾನಾಂತರ + ggml_nbytes(dst), align_ಬೈಟುಗಳು);

    ಆದರೆ (use_vec_reduce) {
        ಸ್ಥಿರಾಂಕ uint64_t tmp_ದತ್ತಾಂಶ_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ಸ್ಥಿರಾಂಕ uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_ತಳ                 = tmp_ದತ್ತಾಂಶ_elems;
        tmp_ಗಾತ್ರ_ಬೈಟುಗಳು =
            ROUNDUP_POW2((tmp_ದತ್ತಾಂಶ_elems + tmp_stats_elems) * ಗಾತ್ರ(ದಶಮಾಂಶ), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_ತಳ <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_ಸ್ಥಾನಾಂತರ = scratch_ಸ್ಥಾನಾಂತರ;
        tmp_bind_ಗಾತ್ರ   = tmp_ಗಾತ್ರ_ಬೈಟುಗಳು;
        scratch_ಸ್ಥಾನಾಂತರ  = ROUNDUP_POW2(scratch_ಸ್ಥಾನಾಂತರ + tmp_ಗಾತ್ರ_ಬೈಟುಗಳು, align_ಬೈಟುಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_ಗಾತ್ರ_ಬೈಟುಗಳು  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_ಸ್ಥಾನಾಂತರ = scratch_ಸ್ಥಾನಾಂತರ;
        tmp_bind_ಗಾತ್ರ   = tmp_ಗಾತ್ರ_ಬೈಟುಗಳು;
        scratch_ಸ್ಥಾನಾಂತರ  = ROUNDUP_POW2(scratch_ಸ್ಥಾನಾಂತರ + tmp_ಗಾತ್ರ_ಬೈಟುಗಳು, align_ಬೈಟುಗಳು);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> blk_entries;
    ಆದರೆ (op.has_ಮುಖವಾಡ) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ಸ್ಥಿರಾಂಕ uint32_t stride_mask3 = (uint32_t) (ಮುಖವಾಡ->nb[3] / ggml_type_ಗಾತ್ರ(ಮುಖವಾಡ->type));
        blk_batch_ಎಣಿಕೆ             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ಸ್ಥಿರಾಂಕ uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_ಎಣಿಕೆ;
        blk_ಗಾತ್ರ_ಬೈಟುಗಳು              = ROUNDUP_POW2(blk_elems * ಗಾತ್ರ(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        ಸ್ಥಿರಾಂಕ ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ಮುಖವಾಡ) / ggml_type_ಗಾತ್ರ(ಮುಖವಾಡ->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(ಮುಖವಾಡ),
                                              ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, ಮುಖವಾಡ),
                                              ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, ಮುಖವಾಡ)),
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, blk_buf, scratch_ಸ್ಥಾನಾಂತರ, blk_ಗಾತ್ರ_ಬೈಟುಗಳು),
        };
        scratch_ಸ್ಥಾನಾಂತರ = ROUNDUP_POW2(scratch_ಸ್ಥಾನಾಂತರ + blk_ಗಾತ್ರ_ಬೈಟುಗಳು, align_ಬೈಟುಗಳು);
    }

    std::vector<uint32_t> ವಿಭಜಿಸು_params = op.params;
    ಆದರೆ (op.has_ಮುಖವಾಡ) {
        ವಿಭಜಿಸು_params.push_back(0u);                     // blk_base
        ವಿಭಜಿಸು_params.push_back(blk_nblk0);              // blk_nblk0
        ವಿಭಜಿಸು_params.push_back(blk_nblk1);              // blk_nblk1
    }
    ವಿಭಜಿಸು_params.push_back(0u);                         // tmp_data_base
    ವಿಭಜಿಸು_params.push_back((uint32_t) tmp_stats_ತಳ);  // tmp_stats_base
    ವಿಭಜಿಸು_params.push_back(nwg);                        // nwg

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> ವಿಭಜಿಸು_entries = {
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, Q),
                                          ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, Q)),
    };
    ಆದರೆ (op.kv_overlap) {
        ವಿಭಜಿಸು_entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(K), op.kv_bind_ಸ್ಥಾನಾಂತರ, op.kv_bind_ಗಾತ್ರ));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ವಿಭಜಿಸು_entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, K),
                                                                  ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, K)));
        ವಿಭಜಿಸು_entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, V),
                                                                  ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, V)));
    }
    uint32_t ವಿಭಜಿಸು_binding_ಸೂಚ್ಯಂಕ = op.kv_overlap ? 2u : 3u;
    ಆದರೆ (op.has_ಮುಖವಾಡ) {
        ವಿಭಜಿಸು_entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(ವಿಭಜಿಸು_binding_ಸೂಚ್ಯಂಕ++, ggml_webgpu_tensor_buf(ಮುಖವಾಡ),
                                                                  ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, ಮುಖವಾಡ),
                                                                  ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, ಮುಖವಾಡ)));
    }
    ಆದರೆ (op.has_sinks) {
        ವಿಭಜಿಸು_entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(ವಿಭಜಿಸು_binding_ಸೂಚ್ಯಂಕ++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, sinks)));
    }
    ಆದರೆ (op.has_ಮುಖವಾಡ) {
        ವಿಭಜಿಸು_entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(ವಿಭಜಿಸು_binding_ಸೂಚ್ಯಂಕ++, blk_buf, blk_entries[1].ಸ್ಥಾನಾಂತರ, blk_ಗಾತ್ರ_ಬೈಟುಗಳು));
    }
    ವಿಭಜಿಸು_entries.push_back(
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(ವಿಭಜಿಸು_binding_ಸೂಚ್ಯಂಕ++, tmp_buf, tmp_bind_ಸ್ಥಾನಾಂತರ, tmp_bind_ಗಾತ್ರ));
    ವಿಭಜಿಸು_entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(ವಿಭಜಿಸು_binding_ಸೂಚ್ಯಂಕ++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, dst),
                                                              ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> reduce_entries;
    ಆದರೆ (use_vec_reduce) {
        ಸ್ಥಿರಾಂಕ uint32_t reduce_sg_ಗಾತ್ರ = ctx->global_ctx->capabilities.max_subgroup_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t reduce_wg_ಗಾತ್ರ = std::max(
            reduce_sg_ಗಾತ್ರ,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_ಗಾತ್ರ,
                                          ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_ಗಾತ್ರ                    = reduce_wg_ಗಾತ್ರ;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_ತಳ,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, tmp_buf, tmp_bind_ಸ್ಥಾನಾಂತರ, tmp_ಗಾತ್ರ_ಬೈಟುಗಳು),
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, dst),
                                              ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ಸ್ಥಿರಾಂಕ uint64_t ವಿಭಜಿಸು_wg_ಒಟ್ಟು = (uint64_t) wg_x * nwg;
    GGML_ASSERT(ವಿಭಜಿಸು_wg_ಒಟ್ಟು <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    ಆದರೆ (op.has_ಮುಖವಾಡ) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_ಎಣಿಕೆ }
        });
    }
    dispatches.push_back({
        pipeline, std::move(ವಿಭಜಿಸು_params), std::move(ವಿಭಜಿಸು_entries), { (uint32_t) ವಿಭಜಿಸು_wg_ಒಟ್ಟು, 1u }
    });
    ಆದರೆ (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    ಮುಖವಾಡ,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, ಮುಖವಾಡ, sinks, dst);
    ಆದರೆ (ggml_webgpu_flash_attn_use_vec_ಮಾರ್ಗ(ctx->global_ctx, Q, K, V)) {
        ಹಿಂತಿರುಗಿ ggml_webgpu_flash_attn_vec(ctx, Q, K, V, ಮುಖವಾಡ, sinks, dst, std::move(op));
    }
    ಹಿಂತಿರುಗಿ ggml_webgpu_flash_attn_ನೇರ(ctx, op);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ತಾರ್ಕಿಕ is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ಶೂನ್ಯಸೂಚಕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    ಆದರೆ (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        ಆಯ್ಕೆ (unary_op) {
            ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    ದಶಮಾಂಶ alpha_n = ggml_get_op_params_f32(dst, 1);
                    ದಶಮಾಂಶ alpha_p = ggml_get_op_params_f32(dst, 2);
                    ದಶಮಾಂಶ beta    = ggml_get_op_params_f32(dst, 3);
                    ದಶಮಾಂಶ eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    ನಿಲ್ಲಿಸು;
                }
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->op == GGML_OP_CLAMP) {
        ದಶಮಾಂಶ clamp_min = ggml_get_op_params_f32(dst, 0);
        ದಶಮಾಂಶ clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->op == GGML_OP_FILL) {
        ದಶಮಾಂಶ fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, effective_src),
    };
    ಆದರೆ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_ದ್ವಿಮಾನ_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_ದ್ವಿಮಾನ_pipeline(shader_lib_ctx, src_overlap);
    ಸ್ವಯಂಚಾಲಿತ *          decisions   = static_cast<ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src0);
    size_t src1_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src1);

    uint32_t ಸ್ಥಾನಾಂತರ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type));
    uint32_t ಸ್ಥಾನಾಂತರ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type));
    size_t   merged_ಸ್ಥಾನಾಂತರ = 0;
    size_t   merged_ಗಾತ್ರ   = 0;
    ಆದರೆ (decisions->src_overlap) {
        ಸ್ಥಿರಾಂಕ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_ಸ್ಥಾನಾಂತರ = merged_range.ಸ್ಥಾನಾಂತರ;
        merged_ಗಾತ್ರ   = merged_range.size;
        ಸ್ಥಾನಾಂತರ_src0   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src0, merged_range);
        ಸ್ಥಾನಾಂತರ_src1   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        ಸ್ಥಾನಾಂತರ_src0,
        ಸ್ಥಾನಾಂತರ_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries;

    ಆದರೆ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src0), merged_ಸ್ಥಾನಾಂತರ, merged_ಗಾತ್ರ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst));
    } ಇಲ್ಲದಿದ್ದರೆ {
        entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ,
                                                            ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ,
                                                            ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src1)));
        ಆದರೆ (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_ಕೂಡಿಸು_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ಕೂಡಿಸು_id_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_ಗಾತ್ರ(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_ಗಾತ್ರ(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, src2));

    ಆದರೆ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       ಒಟ್ಟು_wg       = ggml_nrows(dst);
    ಸ್ಥಿರಾಂಕ uint32_t max_wg_ಪ್ರತಿ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension;
    compute_2d_workgroups(ಒಟ್ಟು_wg, max_wg_ಪ್ರತಿ_dim, wg_x, wg_y);

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    ಆದರೆ (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        ಹಿಂತಿರುಗಿ {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    ಸ್ವಯಂಚಾಲಿತ *          decisions   = static_cast<ggml_webgpu_ದ್ವಿಮಾನ_shader_decisions *>(pipeline.context.get());

    uint32_t ಸ್ಥಾನಾಂತರ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type));
    uint32_t ಸ್ಥಾನಾಂತರ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type));
    size_t   merged_ಸ್ಥಾನಾಂತರ = 0;
    size_t   merged_ಗಾತ್ರ   = 0;
    ಆದರೆ (decisions->src_overlap) {
        ಆದರೆ (ggml_nbytes(src0) == 0) {
            merged_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src1);
            merged_ಗಾತ್ರ   = ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src1);
            ಸ್ಥಾನಾಂತರ_src0   = 0;
            ಸ್ಥಾನಾಂತರ_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_nbytes(src1) == 0) {
            merged_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src0);
            merged_ಗಾತ್ರ   = ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src0);
            ಸ್ಥಾನಾಂತರ_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type));
            ಸ್ಥಾನಾಂತರ_src1   = 0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_ಸ್ಥಾನಾಂತರ = merged_range.ಸ್ಥಾನಾಂತರ;
            merged_ಗಾತ್ರ   = merged_range.size;
            ಸ್ಥಾನಾಂತರ_src0   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src0, merged_range);
            ಸ್ಥಾನಾಂತರ_src1   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     ಸ್ಥಾನಾಂತರ_src0,
                                     ಸ್ಥಾನಾಂತರ_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {};
    ಆದರೆ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src0), merged_ಸ್ಥಾನಾಂತರ, merged_ಗಾತ್ರ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst));
    } ಇಲ್ಲದಿದ್ದರೆ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ne, decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ne, decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    ಆದರೆ (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t ಸ್ಥಾನಾಂತರ_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_ಗಾತ್ರ(rn_src->type));
    uint32_t ಸ್ಥಾನಾಂತರ_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_ಗಾತ್ರ(mul_src->type));
    size_t merged_ಸ್ಥಾನಾಂತರ = 0;
    size_t merged_ಗಾತ್ರ   = 0;

    std::vector<uint32_t> params = {
        ಸ್ಥಾನಾಂತರ_rn_src,
        ಸ್ಥಾನಾಂತರ_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_ಗಾತ್ರ(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_ಗಾತ್ರ(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_ಗಾತ್ರ(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_ಗಾತ್ರ(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_ಗಾತ್ರ(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_ಗಾತ್ರ(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
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

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    ಸ್ವಯಂಚಾಲಿತ *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    ಆದರೆ (decisions->src_overlap) {
        ಸ್ಥಿರಾಂಕ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_ಸ್ಥಾನಾಂತರ  = merged_range.ಸ್ಥಾನಾಂತರ;
        merged_ಗಾತ್ರ    = merged_range.size;
        ಸ್ಥಾನಾಂತರ_rn_src  = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(rn_src, merged_range);
        ಸ್ಥಾನಾಂತರ_mul_src = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(mul_src, merged_range);
        params[0]      = ಸ್ಥಾನಾಂತರ_rn_src;
        params[1]      = ಸ್ಥಾನಾಂತರ_mul_src;
    }

    ಆದರೆ (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, mul_src));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(rn_src), merged_ಸ್ಥಾನಾಂತರ, merged_ಗಾತ್ರ));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst));
    } ಇಲ್ಲದಿದ್ದರೆ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x,
                          wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_ಅಡ್ಡಸಾಲು_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ಅಡ್ಡಸಾಲು_norm_pipeline(shader_lib_ctx);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src) };
    ಆದರೆ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x,
                          wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ inplace         = decisions->inplace;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  has_freq_ಗುಣಕ = (src2 != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims     = ((int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode       = ((int32_t *) dst->op_params)[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs     = ((int32_t *) dst->op_params)[15];

    ದಶಮಾಂಶ freq_ತಳ;
    ದಶಮಾಂಶ freq_scale;
    ದಶಮಾಂಶ ext_ಗುಣಕ;
    ದಶಮಾಂಶ attn_ಗುಣಕ;
    ದಶಮಾಂಶ beta_fast;
    ದಶಮಾಂಶ beta_slow;
    ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ, (int32_t *) dst->op_params + 5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&freq_scale, (int32_t *) dst->op_params + 6, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ext_ಗುಣಕ, (int32_t *) dst->op_params + 7, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&attn_ಗುಣಕ, (int32_t *) dst->op_params + 8, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_fast, (int32_t *) dst->op_params + 9, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_slow, (int32_t *) dst->op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಪೂರ್ಣಾಂಕ sections[4];
    ಸ್ಮರಣೆ_ನಕಲು(sections, (int32_t *) dst->op_params + 11, 4 * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));

    ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    ದಶಮಾಂಶ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)),
        src2 != ಶೂನ್ಯಸೂಚಕ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_ಗುಣಕ),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_ಗುಣಕ),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries     = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    ಆದರೆ (has_freq_ಗುಣಕ) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 2, src2));
    }
    ಆದರೆ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ      src_overlap = src1 != ಶೂನ್ಯಸೂಚಕ && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    ಸ್ವಯಂಚಾಲಿತ * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು = (src1 != ಶೂನ್ಯಸೂಚಕ);

    uint32_t ಸ್ಥಾನಾಂತರ_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type));
    uint32_t ಸ್ಥಾನಾಂತರ_src1 =
        src1 != ಶೂನ್ಯಸೂಚಕ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)) : 0;
    size_t merged_ಸ್ಥಾನಾಂತರ = 0;
    size_t merged_ಗಾತ್ರ   = 0;
    ಆದರೆ (decisions->src_overlap) {
        ಸ್ಥಿರಾಂಕ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_ಸ್ಥಾನಾಂತರ = merged_range.ಸ್ಥಾನಾಂತರ;
        merged_ಗಾತ್ರ   = merged_range.size;
        ಸ್ಥಾನಾಂತರ_src0   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src0, merged_range);
        ಸ್ಥಾನಾಂತರ_src1   = ggml_webgpu_tensor_merged_element_ಸ್ಥಾನಾಂತರ(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ಸ್ಥಾನಾಂತರ_src0,
        ಸ್ಥಾನಾಂತರ_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
        src1 != ಶೂನ್ಯಸೂಚಕ ? (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        src1 != ಶೂನ್ಯಸೂಚಕ ? (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        src1 != ಶೂನ್ಯಸೂಚಕ ? (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  //ಎಲ್ಲೆ
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries;
    uint32_t                          dst_binding = 1;
    ಆದರೆ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(0, ggml_webgpu_tensor_buf(src0), merged_ಸ್ಥಾನಾಂತರ, merged_ಗಾತ್ರ));
    } ಇಲ್ಲದಿದ್ದರೆ {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src0));
    }
    ಆದರೆ (ವಿಭಜಿಸು && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ಶೂನ್ಯಸೂಚಕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src) };

    ಆದರೆ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ inplace     = decisions->inplace;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  has_ಮುಖವಾಡ    = (src1 != ಶೂನ್ಯಸೂಚಕ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  has_sink    = (src2 != ಶೂನ್ಯಸೂಚಕ);
    ದಶಮಾಂಶ      max_bias    = ggml_get_op_params_f32(dst, 1);
    ದಶಮಾಂಶ      n_ತಲೆ_log2 = ದಶಮಾಂಶ(1u << (uint32_t) floor(log2(src0->ne[2])));
    ದಶಮಾಂಶ      m0          = powf(2.0f, -(max_bias) / n_ತಲೆ_log2);
    ದಶಮಾಂಶ      m1          = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_ಗಾತ್ರ(src0->type)),
        has_ಮುಖವಾಡ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_ಗಾತ್ರ(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_ಗಾತ್ರ(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type)),
        has_ಮುಖವಾಡ ? (uint32_t) (src1->nb[1] / ggml_type_ಗಾತ್ರ(src1->type)) : 0,
        has_ಮುಖವಾಡ ? (uint32_t) (src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type)) : 0,
        has_ಮುಖವಾಡ ? (uint32_t) (src1->nb[3] / ggml_type_ಗಾತ್ರ(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_ಮುಖವಾಡ ? (uint32_t) src1->ne[2] : 0,
        has_ಮುಖವಾಡ ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_ತಲೆ_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries     = { ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src0),
        ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src0)) };
    uint32_t                          binding_ಅಂಕೆ = 1;
    ಆದರೆ (has_ಮುಖವಾಡ) {
        entries.push_back(ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(binding_ಅಂಕೆ, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, src1),
                                                            ggml_webgpu_tensor_binding_ಗಾತ್ರ(ctx, src1)));
        binding_ಅಂಕೆ++;
    }
    ಆದರೆ (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ, src2));
        binding_ಅಂಕೆ++;
    }
    ಆದರೆ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, binding_ಅಂಕೆ, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = ggml_nrows(dst);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ತಾರ್ಕಿಕ is_ತುದಿ_k = dst->op == GGML_OP_ತುದಿ_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ಶೂನ್ಯಸೂಚಕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
    shader_lib_ctx.wg_mem_ಎಲ್ಲೆ_ಬೈಟುಗಳು = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_ಗಾತ್ರ;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    ಸ್ವಯಂಚಾಲಿತ * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_ವಿಲೀನ_pipeline = ctx->shader_lib->get_argsort_ವಿಲೀನ_pipeline(shader_lib_ctx);

    ಸ್ಥಿರಾಂಕ uint32_t src_ne0 = (uint32_t) src->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ಸ್ಥಿರಾಂಕ uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 =
        is_ತುದಿ_k ? std::min(argsort_decisions->wg_ಗಾತ್ರ, (uint32_t) dst->ne[0]) : argsort_decisions->wg_ಗಾತ್ರ;
    uint32_t ಹೊರಹೋಗು_ne0 = src_ne0;
    ಆದರೆ (is_ತುದಿ_k) {
        ಆದರೆ (npr > 1) {
            ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆಯ_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_ಗಾತ್ರ;
            ಹೊರಹೋಗು_ne0                  = (npr - 1) * ಖಂಡ_ಗಾತ್ರ_3 + std::min(ಕೊನೆಯ_tile, ಖಂಡ_ಗಾತ್ರ_3);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹೊರಹೋಗು_ne0 = ಖಂಡ_ಗಾತ್ರ_3;
        }
    }

    uint32_t ವಿಲೀನ_len    = ಖಂಡ_ಗಾತ್ರ_3;
    uint32_t ವಿಲೀನ_passes = 0;
    ಆಗ (ವಿಲೀನ_len < ಹೊರಹೋಗು_ne0) {
        ವಿಲೀನ_len <<= 1;
        ವಿಲೀನ_passes++;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಆರಂಭ_ಒಳ_tmp = (ವಿಲೀನ_passes % 2) == 1;

    ಸ್ಥಿರಾಂಕ size_t dst_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(dst);
    ಸ್ಥಿರಾಂಕ size_t idx_nbytes = ಹೊರಹೋಗು_ne0 * ggml_nrows(dst) * ಗಾತ್ರ(int32_t);
    ಸ್ಥಿರಾಂಕ size_t tmp_ಸ್ಥಾನಾಂತರ =
        ROUNDUP_POW2(dst_ಸ್ಥಾನಾಂತರ + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
    ಸ್ಥಿರಾಂಕ size_t tmp_binding_ಗಾತ್ರ = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ಸ್ಥಿರಾಂಕ size_t dst_binding_ಗಾತ್ರ =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type));
    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type));
    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_tmp  = 0;
    ಸ್ಥಿರಾಂಕ uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type));
    ಸ್ಥಿರಾಂಕ uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type));
    ಸ್ಥಿರಾಂಕ uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type));
    ಸ್ಥಿರಾಂಕ uint32_t stride_idx1 = ಹೊರಹೋಗು_ne0;
    ಸ್ಥಿರಾಂಕ uint32_t stride_idx2 = ಹೊರಹೋಗು_ne0 * (uint32_t) dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ಸ್ಥಿರಾಂಕ uint32_t init_ಸ್ಥಾನಾಂತರ       = ಆರಂಭ_ಒಳ_tmp ? ಸ್ಥಾನಾಂತರ_tmp : ಸ್ಥಾನಾಂತರ_dst;
    ಸ್ಥಿರಾಂಕ size_t   init_align_ಸ್ಥಾನಾಂತರ = ಆರಂಭ_ಒಳ_tmp ? tmp_ಸ್ಥಾನಾಂತರ : ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, dst);
    ಸ್ಥಿರಾಂಕ size_t   init_binding_ಗಾತ್ರ = ಆರಂಭ_ಒಳ_tmp ? tmp_binding_ಗಾತ್ರ : dst_binding_ಗಾತ್ರ;

    std::vector<uint32_t> init_params = {
        ಸ್ಥಾನಾಂತರ_src,  init_ಸ್ಥಾನಾಂತರ, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], ಹೊರಹೋಗು_ne0,
        ಖಂಡ_ಗಾತ್ರ_3,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_wg_init  = npr * nrows;
    ಸ್ಥಿರಾಂಕ uint32_t max_wg_ಪ್ರತಿ_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension;
    compute_2d_workgroups(ಒಟ್ಟು_wg_init, max_wg_ಪ್ರತಿ_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> init_entries = {
        ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
        ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(dst), init_align_ಸ್ಥಾನಾಂತರ, init_binding_ಗಾತ್ರ)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    ಆದರೆ (ವಿಲೀನ_passes == 0) {
        ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    ತಾರ್ಕಿಕ     ಒಳ_is_tmp = ಆರಂಭ_ಒಳ_tmp;
    uint32_t len       = ಖಂಡ_ಗಾತ್ರ_3;
    ಆಗ (len < ಹೊರಹೋಗು_ne0) {
        ಸ್ಥಿರಾಂಕ uint32_t nm = CEIL_DIV(ಹೊರಹೋಗು_ne0, 2 * len);

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ     ಹೊರಹೋಗು_is_tmp  = !ಒಳ_is_tmp;
        ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_ಒಳ   = ಒಳ_is_tmp ? ಸ್ಥಾನಾಂತರ_tmp : ಸ್ಥಾನಾಂತರ_dst;
        ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_ಹೊರಹೋಗು  = ಹೊರಹೋಗು_is_tmp ? ಸ್ಥಾನಾಂತರ_tmp : ಸ್ಥಾನಾಂತರ_dst;
        ಸ್ಥಿರಾಂಕ size_t   align_ಒಳ    = ಒಳ_is_tmp ? tmp_ಸ್ಥಾನಾಂತರ : ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, dst);
        ಸ್ಥಿರಾಂಕ size_t   align_ಹೊರಹೋಗು   = ಹೊರಹೋಗು_is_tmp ? tmp_ಸ್ಥಾನಾಂತರ : ggml_webgpu_tensor_align_ಸ್ಥಾನಾಂತರ(ctx, dst);
        ಸ್ಥಿರಾಂಕ size_t   ಗಾತ್ರ_ಒಳ     = ಒಳ_is_tmp ? tmp_binding_ಗಾತ್ರ : dst_binding_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ size_t   ಗಾತ್ರ_ಹೊರಹೋಗು    = ಹೊರಹೋಗು_is_tmp ? tmp_binding_ಗಾತ್ರ : dst_binding_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t ತುದಿ_k_ಹೊರಹೋಗು   = (is_ತುದಿ_k && nm == 1) ? (uint32_t) dst->ne[0] : ಹೊರಹೋಗು_ne0;
        ಸ್ಥಿರಾಂಕ uint32_t stride_out1 = ತುದಿ_k_ಹೊರಹೋಗು;
        ಸ್ಥಿರಾಂಕ uint32_t stride_out2 = ತುದಿ_k_ಹೊರಹೋಗು * (uint32_t) dst->ne[1];
        ಸ್ಥಿರಾಂಕ uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> ವಿಲೀನ_params = { ಸ್ಥಾನಾಂತರ_src,
                                               ಸ್ಥಾನಾಂತರ_ಒಳ,
                                               ಸ್ಥಾನಾಂತರ_ಹೊರಹೋಗು,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               ಹೊರಹೋಗು_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               ತುದಿ_k_ಹೊರಹೋಗು,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> ವಿಲೀನ_entries = {
            ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(1, ggml_webgpu_tensor_buf(dst), align_ಒಳ, ಗಾತ್ರ_ಒಳ),
            ggml_webgpu_make_bind_ಗುಂಪು_ನಮೂದು(2, ggml_webgpu_tensor_buf(dst), align_ಹೊರಹೋಗು, ಗಾತ್ರ_ಹೊರಹೋಗು)
        };

        uint32_t       wg_x_ವಿಲೀನ;
        uint32_t       wg_y_ವಿಲೀನ;
        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_wg_ವಿಲೀನ = nm * nrows;
        compute_2d_workgroups(ಒಟ್ಟು_wg_ವಿಲೀನ, max_wg_ಪ್ರತಿ_dim, wg_x_ವಿಲೀನ, wg_y_ವಿಲೀನ);

        dispatches.push_back({
            argsort_ವಿಲೀನ_pipeline, std::move(ವಿಲೀನ_params), std::move(ವಿಲೀನ_entries), { wg_x_ವಿಲೀನ, wg_y_ವಿಲೀನ }
        });

        len <<= 1;
        ಒಳ_is_tmp = !ಒಳ_is_tmp;
    }

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = ಶೂನ್ಯಸೂಚಕ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        ಒಟ್ಟು_wg = ggml_nrows(dst);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ತಾರ್ಕಿಕ                  ಒಟ್ಟು_ಮೊತ್ತ = dst->op == GGML_OP_ಮೊತ್ತ;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),
                                     ಒಟ್ಟು_ಮೊತ್ತ ? 0 : (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)),
                                     ಒಟ್ಟು_ಮೊತ್ತ ? 0 : (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)),
                                     ಒಟ್ಟು_ಮೊತ್ತ ? 0 : (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)),
                                     ಒಟ್ಟು_ಮೊತ್ತ ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     ಒಟ್ಟು_ಮೊತ್ತ ? 1 : (uint32_t) src->ne[1],
                                     ಒಟ್ಟು_ಮೊತ್ತ ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = ಒಟ್ಟು_ಮೊತ್ತ ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_webgpu_can_fuse_rms_norm_mul(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // additional constraints specific to this fusion
    ಸ್ಥಿರಾಂಕ ggml_tensor * rms_norm = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul      = cgraph->nodes[ಗಂಟು_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    ಆದರೆ (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    ಆದರೆ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // rms_norm shader assumes contiguous rows
    ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(mul->src[0]) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(mul->src[1])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_webgpu_can_fuse_gdn_cache(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಪೂರ್ಣಾಂಕ & ಅಂಕೆ_encoded_ops) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * gdn = cgraph->nodes[ಗಂಟು_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ಆದರೆ (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * src_v     = gdn->src[2];
    ಸ್ಥಿರಾಂಕ int64_t       S_v       = src_v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t       H         = src_v->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t       n_ಟೋಕನುಗಳು  = src_v->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t       n_seqs    = src_v->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t       D         = S_v * S_v * H;
    ಸ್ಥಿರಾಂಕ int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    ಸ್ಥಿರಾಂಕ int64_t       n_written = std::min<int64_t>(n_ಟೋಕನುಗಳು, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ಸ್ಥಿರಾಂಕ size_t ಬಾಲ_off = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(GGML_TYPE_F32, S_v * H * n_ಟೋಕನುಗಳು * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ಸ್ಥಿರಾಂಕ ggml_tensor * cpy     = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ                 cpy_idx = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಗಂಟು_idx + 1; j < cgraph->n_nodes && cpy == ಶೂನ್ಯಸೂಚಕ; ++j) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * n = cgraph->nodes[j];
        ಆದರೆ (ggml_op_is_ಖಾಲಿಗಳು(n->op) || ggml_is_ಖಾಲಿಗಳು(n)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        cpy     = n;
        cpy_idx = j;
    }
    ಆದರೆ (cpy == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ಸ್ಥಿರಾಂಕ ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ಆದರೆ (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != ಬಾಲ_off ||
        !ggml_is_contiguous(cpy_src)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ಸ್ಥಿರಾಂಕ std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    ಆದರೆ (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(GGML_TYPE_F32, D)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಅಂಕೆ_encoded_ops = cpy_idx - ಗಂಟು_idx + 1;

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t        ಬಹುಲಕ_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_ಗಾತ್ರ(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_ಗಾತ್ರ(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     ಬಹುಲಕ_flags };

    std::vector<wgpu::Bind_ಗುಂಪು_ನಮೂದು> entries = { ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ಗುಂಪು_ನಮೂದು(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_ಗಾತ್ರ = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    ಸ್ವಯಂಚಾಲಿತ *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ಒಟ್ಟು_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_ಗಾತ್ರ);
    compute_2d_workgroups(ಒಟ್ಟು_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension, wg_x, wg_y);

    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
ಸ್ಥಿರ std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           ಪೂರ್ಣಾಂಕ            ಗಂಟು_idx,
                                                           ಪೂರ್ಣಾಂಕ &          ಅಂಕೆ_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  ಗಂಟು  = nodes[ಗಂಟು_idx];

    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(ಗಂಟು)) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << ಗಂಟು << ", " << ggml_op_name(ಗಂಟು->op) << ")");

    ggml_tensor * src0 = ಗಂಟು->src[0];
    ggml_tensor * src1 = ಗಂಟು->src[1];
    ggml_tensor * src2 = ಗಂಟು->src[2];

    ಆಯ್ಕೆ (ಗಂಟು->op) {
            // no-ops
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
            ಹಿಂತಿರುಗಿ std::nullopt;
        ಸ್ಥಿತಿ GGML_OP_DUP:
        ಸ್ಥಿತಿ GGML_OP_CPY:
        ಸ್ಥಿತಿ GGML_OP_CONT:
            ಹಿಂತಿರುಗಿ ggml_webgpu_cpy(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ಗಣ(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ ggml_webgpu_get_ಅಡ್ಡಸಾಲುಗಳು(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಹಿಂತಿರುಗಿ ggml_webgpu_mul_mat(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            ಹಿಂತಿರುಗಿ ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ಹಿಂತಿರುಗಿ ggml_webgpu_flash_attn(ctx, src0, src1, src2, ಗಂಟು->src[3], ಗಂಟು->src[4], ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_DIV:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ದ್ವಿಮಾನ_op(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ಕೂಡಿಸು_id(ctx, src0, src1, src2, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            ಹಿಂತಿರುಗಿ ggml_webgpu_concat(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            ಹಿಂತಿರುಗಿ ggml_webgpu_repeat(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಆದರೆ (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, ಗಂಟು_idx)) {
                ಅಂಕೆ_encoded_ops        = 2;
                ggml_tensor * mul_ಗಂಟು = nodes[ಗಂಟು_idx + 1];
                ಹಿಂತಿರುಗಿ ggml_webgpu_rms_norm_mul(ctx, src0, ಗಂಟು, mul_ಗಂಟು->src[0], mul_ಗಂಟು->src[1], mul_ಗಂಟು);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ ggml_webgpu_ಅಡ್ಡಸಾಲು_norm(ctx, src0, ಗಂಟು);
            }
        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ಅಡ್ಡಸಾಲು_norm(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            ಹಿಂತಿರುಗಿ ggml_webgpu_rope(ctx, src0, src1, src2, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಹಿಂತಿರುಗಿ ggml_webgpu_glu(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            ಹಿಂತಿರುಗಿ ggml_webgpu_scale(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            ಹಿಂತಿರುಗಿ ggml_webgpu_soft_max(ctx, src0, src1, src2, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_UNARY:
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಸ್ಥಿತಿ GGML_OP_FILL:
        ಸ್ಥಿತಿ GGML_OP_LOG:
        ಸ್ಥಿತಿ GGML_OP_SQR:
        ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಸ್ಥಿತಿ GGML_OP_SIN:
        ಸ್ಥಿತಿ GGML_OP_COS:
        ಸ್ಥಿತಿ GGML_OP_DIAG:
        ಸ್ಥಿತಿ GGML_OP_TRI:
            ಹಿಂತಿರುಗಿ ggml_webgpu_unary_op(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            ಹಿಂತಿರುಗಿ ggml_webgpu_solve_tri(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ssm_conv(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ssm_ಪರಿಶೀಲನೆ(ctx, src0, src1, src2, ಗಂಟು->src[3], ಗಂಟು->src[4], ಗಂಟು->src[5], ಗಂಟು->src[6],
                                        ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            ಆದರೆ (ggml_webgpu_can_fuse_gdn_cache(cgraph, ಗಂಟು_idx, ಅಂಕೆ_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[ಗಂಟು_idx + ಅಂಕೆ_encoded_ops - 1]->src[1];
                ಹಿಂತಿರುಗಿ ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ಗಂಟು->src[3], ಗಂಟು->src[4], ಗಂಟು->src[5],
                                                   ಗಂಟು, dst_fuse);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ಗಂಟು->src[3], ಗಂಟು->src[4], ಗಂಟು->src[5],
                                                   ಗಂಟು, ಶೂನ್ಯಸೂಚಕ);
            }
        ಸ್ಥಿತಿ GGML_OP_PAD:
            ಹಿಂತಿರುಗಿ ggml_webgpu_pad(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            ಹಿಂತಿರುಗಿ ggml_webgpu_argmax(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            // we reuse the same argsort implementation for top_k
            ಹಿಂತಿರುಗಿ ggml_webgpu_argsort(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            ಹಿಂತಿರುಗಿ ggml_webgpu_cumsum(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ ggml_webgpu_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ctx, src0, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            ಹಿಂತಿರುಗಿ ggml_webgpu_conv_2d(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
            ಹಿಂತಿರುಗಿ ggml_webgpu_conv_2d_dw(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            ಹಿಂತಿರುಗಿ ggml_webgpu_im2col(ctx, src0, src1, ಗಂಟು);
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            ಹಿಂತಿರುಗಿ ggml_webgpu_upscale(ctx, src0, ಗಂಟು);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ std::nullopt;
    }
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ಸ್ಥಿರಾಂಕ std::vector<std::string> & pipeline_ಹೆಸರುಗಳು,
                                                        uint32_t &                       ಅಂಕೆ_inflight_batches) {
    ಆದರೆ (pipeline_ಹೆಸರುಗಳು.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Resolve_Query_ಗಣ(ctx->profile_timestamp_query_ಗಣ, 0, ctx->profile_timestamp_query_ಎಣಿಕೆ,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_ಬಫರ್_ಗಮ್ಯ_ಬಫರ್(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_ಎಣಿಕೆ * ಗಾತ್ರ(uint64_t));

    wgpu::Command_ಬಫರ್ profile_commands = encoder.ಮುಕ್ತಾಯ();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, ಅಂಕೆ_inflight_batches);

    ಸ್ಥಿರಾಂಕ size_t mapped_ಗಾತ್ರ = ctx->profile_timestamp_query_ಎಣಿಕೆ * ಗಾತ್ರ(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_ಎಣಿಕೆ == 2 * pipeline_ಹೆಸರುಗಳು.size());

    ggml_backend_webgpu_map_ಬಫರ್(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::Map_ಬಹುಲಕ::Read, 0,
                                   mapped_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ uint64_t * ts_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_ಗಾತ್ರ);

    ಗಾಗಿ (size_t i = 0; i < pipeline_ಹೆಸರುಗಳು.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ elapsed_ms = ದ್ವಿಗುಣ(ts_ದತ್ತಾಂಶ[2 * i + 1] - ts_ದತ್ತಾಂಶ[2 * i]) * 1e-6;
        ctx->shader_gpu_ಸಮಯ_ms[pipeline_ಹೆಸರುಗಳು[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#ಆದರೆ_ಅಂತ್ಯ

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_check_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(webgpu_context & ctx, uint32_t & ಅಂಕೆ_inflight_batches) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_CHECK_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_ಬಫರ್_ಗಮ್ಯ_ಬಫರ್(ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf, 0, ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf, 0,
                               ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf.Get_ಗಾತ್ರ());
    wgpu::Command_ಬಫರ್ commands = encoder.ಮುಕ್ತಾಯ();
    ggml_backend_webgpu_submit_commands(ctx, commands, ಅಂಕೆ_inflight_batches);
    ggml_backend_webgpu_map_ಬಫರ್(ctx->global_ctx, ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf, wgpu::Map_ಬಹುಲಕ::Read, 0,
                                   ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf.Get_ಗಾತ್ರ());
    ಸ್ಥಿರಾಂಕ uint32_t * error_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint32_t *) ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf.GetConstMappedRange();
    ಆದರೆ (*error_ದತ್ತಾಂಶ) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf.Unmap();
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(ಅಂಕೆ_inflight_batches);
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_webgpu_ನಕ್ಷೆ_compute(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(ನಕ್ಷೆ_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t ಅಂಕೆ_batched_kernels  = 0;
    uint32_t ಅಂಕೆ_inflight_batches = 0;
    ತಾರ್ಕಿಕ     ಒಳಗೊಂಡಿದೆ_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು    = ಸುಳ್ಳು;
    ಪೂರ್ಣಾಂಕ      ಅಂಕೆ_encoded_ops      = 1;
    ಪೂರ್ಣಾಂಕ      ಗಂಟು_idx             = 0;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_ಎಣಿಕೆ = 0;
    std::vector<std::string> profile_pipeline_ಹೆಸರುಗಳು;
#ಆದರೆ_ಅಂತ್ಯ

    ctx->ಸಕ್ರಿಯ_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    ಆದರೆ (ctx->batch_compute_passes) {
        ctx->ಸಕ್ರಿಯ_compute_pass = ctx->ಸಕ್ರಿಯ_command_encoder.BeginComputePass();
    }

    ಆಗ (ಗಂಟು_idx < cgraph->n_nodes) {
        ಆದರೆ (cgraph->nodes[ಗಂಟು_idx]->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
            ಒಳಗೊಂಡಿದೆ_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು = ನಿಜ;
        }
        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ cmd = ggml_webgpu_encode(ctx, cgraph, ಗಂಟು_idx, ಅಂಕೆ_encoded_ops)) {
            commands.push_back(*cmd);
            ಅಂಕೆ_batched_kernels += cmd.value().ಅಂಕೆ_kernels;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_ಹೆಸರುಗಳು.insert(profile_pipeline_ಹೆಸರುಗಳು.end(), cmd->pipeline_ಹೆಸರುಗಳು.begin(),
                                          cmd->pipeline_ಹೆಸರುಗಳು.end());
#ಆದರೆ_ಅಂತ್ಯ
        }

        ಆದರೆ (ಅಂಕೆ_batched_kernels >= ctx->global_ctx->command_submit_batch_ಗಾತ್ರ) {
            ಆದರೆ (ctx->ಸಕ್ರಿಯ_compute_pass) {
                ctx->ಸಕ್ರಿಯ_compute_pass.ಕೊನೆ_2();
            }
            ಅಂಕೆ_batched_kernels                = 0;
            wgpu::Command_ಬಫರ್ batch_commands = ctx->ಸಕ್ರಿಯ_command_encoder.ಮುಕ್ತಾಯ();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, ಅಂಕೆ_inflight_batches);

            // reset state for next batch
            ctx->ಸಕ್ರಿಯ_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            ಆದರೆ (ctx->batch_compute_passes) {
                ctx->ಸಕ್ರಿಯ_compute_pass = ctx->ಸಕ್ರಿಯ_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            ಆದರೆ (ctx->profile_timestamp_query_ಎಣಿಕೆ + 2 * ctx->global_ctx->command_submit_batch_ಗಾತ್ರ >=
                WEBGPU_MAX_PROFILE_QUERY_ಎಣಿಕೆ) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_ಹೆಸರುಗಳು, ಅಂಕೆ_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_ಎಣಿಕೆ = 0;
                profile_pipeline_ಹೆಸರುಗಳು.clear();
            }
#ಆದರೆ_ಅಂತ್ಯ
        }

        ಗಂಟು_idx += ಅಂಕೆ_encoded_ops;
        ಅಂಕೆ_encoded_ops = 1;
    }

    ಆದರೆ (ctx->ಸಕ್ರಿಯ_compute_pass) {
        ctx->ಸಕ್ರಿಯ_compute_pass.ಕೊನೆ_2();
        ctx->ಸಕ್ರಿಯ_compute_pass = ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (ಅಂಕೆ_batched_kernels > 0) {
        wgpu::Command_ಬಫರ್ batch_commands = ctx->ಸಕ್ರಿಯ_command_encoder.ಮುಕ್ತಾಯ();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, ಅಂಕೆ_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->ಸಕ್ರಿಯ_command_encoder = ಶೂನ್ಯಸೂಚಕ;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_ಹೆಸರುಗಳು, ಅಂಕೆ_inflight_batches);
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (ಒಳಗೊಂಡಿದೆ_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
        ggml_backend_webgpu_check_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, ಅಂಕೆ_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(ನಕ್ಷೆ_compute, ctx->global_ctx);
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ರಚನೆ ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    ತಾರ್ಕಿಕ                  recorded = ಸುಳ್ಳು;
};

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    ಸ್ವಯಂಚಾಲಿತ * event_ctx      = ಹೊಸ ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    ಸ್ವಯಂಚಾಲಿತ * event   = ಹೊಸ ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    ಹಿಂತಿರುಗಿ event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_ಬಳಸದ(dev);
    ಅಳಿಸು static_cast<ggml_backend_webgpu_event_context *>(event->context);
    ಅಳಿಸು event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_ಬಳಸದ(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    ಆದರೆ (!event_ctx->recorded) {
        ಹಿಂತಿರುಗಿ;
    }
    wgpu::Wait_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    ಆದರೆ (ಸ್ಥಿತಿ_2 == wgpu::Wait_ಸ್ಥಿತಿ::Timed_ಹೊರಹೋಗು) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_event_ದಾಖಲೆ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.On_Submitted_Work_ಮುಗಿದ(
        wgpu::Callback_ಬಹುಲಕ::AllowSpontaneous, [](wgpu::ಸರತಿ_Work_ಮುಗಿದ_ಸ್ಥಿತಿ, wgpu::StringView) {});
    event_ctx->recorded = ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ಬಳಸದ(backend);
    ggml_backend_webgpu_device_event_synchronize(ಶೂನ್ಯಸೂಚಕ, event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_ಗಣ_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *   ದತ್ತಾಂಶ,
                                                 size_t         ಸ್ಥಾನಾಂತರ,
                                                 size_t         size) {
    GGML_ಬಳಸದ(backend);
    ಸ್ವಯಂಚಾಲಿತ * buf_ctx      = (ggml_backend_webgpu_ಬಫರ್_context *) tensor->ಬಫರ್->context;
    size_t ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(tensor) + ಸ್ಥಾನಾಂತರ;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_ಬಫರ್(buf_ctx->ಬಫರ್, ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, (size / 4) * 4);

    ಆದರೆ (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t ಉಳಿದ_ಗಾತ್ರ = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        ಗಾಗಿ (size_t i = 0; i < ಉಳಿದ_ಗಾತ್ರ; i++) {
            ((uint8_t *) &val32)[i] = ((ಸ್ಥಿರಾಂಕ uint8_t *) ದತ್ತಾಂಶ)[size - ಉಳಿದ_ಗಾತ್ರ + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_ಬಫರ್_memset(buf_ctx->global_ctx, buf_ctx->ಬಫರ್, val32,
                                          ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ + (size - ಉಳಿದ_ಗಾತ್ರ), ಉಳಿದ_ಗಾತ್ರ);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_ಸರತಿ(backend_ctx->webgpu_ctx->global_ctx);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ಖಾಲಿ,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ಖಾಲಿ,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_webgpu_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_ದಾಖಲೆ,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_webgpu_ಬಫರ್_context * ctx = static_cast<ggml_backend_webgpu_ಬಫರ್_context *>(ಬಫರ್->context);
    ಆದರೆ (ctx != ಶೂನ್ಯಸೂಚಕ && ctx->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
        ctx->ಬಫರ್.Destroy();
        ಅಳಿಸು ctx;
    }
}

// Returns the "fake" base pointer.
ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_webgpu_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ಬಳಸದ(ಬಫರ್);
    ಹಿಂತಿರುಗಿ webgpu_ptr_ತಳ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                ಸ್ಥಾನಾಂತರ,
                                                     size_t                size) {
    ಆದರೆ (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        ಹಿಂತಿರುಗಿ;
    }

    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(memset_tensor);

    ggml_backend_webgpu_ಬಫರ್_context * buf_ctx = (ggml_backend_webgpu_ಬಫರ್_context *) ಬಫರ್->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ")");

    size_t ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(tensor) + ಸ್ಥಾನಾಂತರ;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_ಬಫರ್_memset(buf_ctx->global_ctx, buf_ctx->ಬಫರ್, val32, ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ, size);
    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(memset_tensor, buf_ctx->global_ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                  ggml_tensor *         tensor,
                                                  ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                  size_t                ಸ್ಥಾನಾಂತರ,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(ಗಣ_tensor);
    ggml_backend_webgpu_ಬಫರ್_context * buf_ctx = (ggml_backend_webgpu_ಬಫರ್_context *) ಬಫರ್->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << ದತ್ತಾಂಶ
                                                              << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ")");

    size_t ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(tensor) + ಸ್ಥಾನಾಂತರ;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    ಆದರೆ (ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ % 4 != 0) {
        size_t lane = ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t ತಲೆ = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t ತಲೆ_val = 0;
        ಗಾಗಿ (size_t i = 0; i < ತಲೆ; i++) {
            ((uint8_t *) &ತಲೆ_val)[lane + i] = ((ಸ್ಥಿರಾಂಕ uint8_t *) ದತ್ತಾಂಶ)[i];
        }
        ggml_backend_webgpu_ಬಫರ್_memset(buf_ctx->global_ctx, buf_ctx->ಬಫರ್, ತಲೆ_val, ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ, ತಲೆ);

        ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ += ತಲೆ;
        size -= ತಲೆ;
        ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint8_t *) ದತ್ತಾಂಶ + ತಲೆ;
    }

    ಆದರೆ (size > 0) {
        buf_ctx->global_ctx->queue.Write_ಬಫರ್(buf_ctx->ಬಫರ್, ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, (size / 4) * 4);
    }

    ಆದರೆ (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t ಉಳಿದ_ಗಾತ್ರ = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        ಗಾಗಿ (size_t i = 0; i < ಉಳಿದ_ಗಾತ್ರ; i++) {
            ((uint8_t *) &val32)[i] = ((ಸ್ಥಿರಾಂಕ uint8_t *) ದತ್ತಾಂಶ)[size - ಉಳಿದ_ಗಾತ್ರ + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_ಬಫರ್_memset(buf_ctx->global_ctx, buf_ctx->ಬಫರ್, val32,
                                          ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ + (size - ಉಳಿದ_ಗಾತ್ರ), ಉಳಿದ_ಗಾತ್ರ);
    }
    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(ಗಣ_tensor, buf_ctx->global_ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                  ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                                  ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                                  size_t                ಸ್ಥಾನಾಂತರ,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(get_tensor);
    ggml_backend_webgpu_ಬಫರ್_context * buf_ctx = (ggml_backend_webgpu_ಬಫರ್_context *) ಬಫರ್->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << ದತ್ತಾಂಶ
                                                              << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ = ggml_webgpu_tensor_ಸ್ಥಾನಾಂತರ(tensor) + ಸ್ಥಾನಾಂತರ;

    size_t local_ಸ್ಥಾನಾಂತರ = ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ % 4;
    ಆದರೆ (local_ಸ್ಥಾನಾಂತರ != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ -= local_ಸ್ಥಾನಾಂತರ;
    }

    size_t final_ಗಾತ್ರ = size + local_ಸ್ಥಾನಾಂತರ;
    ಆದರೆ (final_ಗಾತ್ರ % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_ಗಾತ್ರ += 4 - (final_ಗಾತ್ರ % 4);
    }

    std::lock_guard<std::recursive_mutex> ಬೀಗ(buf_ctx->global_ctx->mutex);

    ಆದರೆ (buf_ctx->global_ctx->get_tensor_staging_buf == ಶೂನ್ಯಸೂಚಕ ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_ಗಾತ್ರ() < final_ಗಾತ್ರ) {
        // Create a new staging buffer if it doesn't exist or is too small
        ಆದರೆ (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_ಬಫರ್(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_ಗಾತ್ರ,
                                  wgpu::ಬಫರ್_Usage::CopyDst | wgpu::ಬಫರ್_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_ಬಫರ್_ಗಮ್ಯ_ಬಫರ್(buf_ctx->ಬಫರ್, ಒಟ್ಟು_ಸ್ಥಾನಾಂತರ, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_ಗಾತ್ರ);
    wgpu::Command_ಬಫರ್ commands = encoder.ಮುಕ್ತಾಯ();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_ಬಫರ್(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::Map_ಬಹುಲಕ::Read, 0, final_ಗಾತ್ರ);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_ಗಾತ್ರ);

    // Copy the data from the mapped range to the output buffer
    std::ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) mapped_range + local_ಸ್ಥಾನಾಂತರ), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(get_tensor, buf_ctx->global_ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << ಬಫರ್ << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(clear);
    ggml_backend_webgpu_ಬಫರ್_context * buf_ctx = (ggml_backend_webgpu_ಬಫರ್_context *) ಬಫರ್->context;
    ggml_backend_webgpu_ಬಫರ್_memset(buf_ctx->global_ctx, buf_ctx->ಬಫರ್, value, 0, ಬಫರ್->size);
    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(clear, buf_ctx->global_ctx);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_webgpu_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_webgpu_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ಖಾಲಿ,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ಖಾಲಿ,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_webgpu_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ಹಿಂತಿರುಗಿ ctx->device_name.c_str();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_webgpu_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft,
                                                                          size_t                     size) {
    ಸ್ಥಿರ std::atomic<ಪೂರ್ಣಾಂಕ> ಬಫರ್_ಎಣಿಕೆ;
    ಪೂರ್ಣಾಂಕ                     ಬಫರ್_id = ಬಫರ್_ಎಣಿಕೆ++;
    std::string             buf_name  = "tensor_buf" + std::to_string(ಬಫರ್_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << ಬಫರ್_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::ಬಫರ್_2                         buf;
    ggml_webgpu_create_ಬಫರ್(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::ಬಫರ್_Usage::Storage | wgpu::ಬಫರ್_Usage::CopySrc | wgpu::ಬಫರ್_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_ಬಫರ್_context * buf_ctx =
        ಹೊಸ ggml_backend_webgpu_ಬಫರ್_context(buf, buf_name, ctx->webgpu_global_ctx);

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_webgpu_ಬಫರ್_interface, buf_ctx, size);
}

ಸ್ಥಿರ size_t ggml_backend_webgpu_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ಹಿಂತಿರುಗಿ dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
ಸ್ಥಿರ size_t ggml_backend_webgpu_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ಹಿಂತಿರುಗಿ dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ;
}

ಸ್ಥಿರ size_t ggml_backend_webgpu_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft,
                                                             ಸ್ಥಿರಾಂಕ ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    ಆಯ್ಕೆ (tensor->op) {
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = tensor->src[0];
                ಆದರೆ (src0) {
                    ಸ್ಥಿರಾಂಕ size_t ತುಂಬಿದ = ಗಾತ್ರ(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        ತುಂಬಿದ * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
                ಸ್ಥಿರಾಂಕ ggml_tensor * Q            = tensor->src[0];
                ಸ್ಥಿರಾಂಕ ggml_tensor * K            = tensor->src[1];
                ಸ್ಥಿರಾಂಕ ggml_tensor * V            = tensor->src[2];
                ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ         = tensor->src[3];
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &        capabilities = ctx->webgpu_global_ctx->capabilities;
                ಆದರೆ (ggml_webgpu_flash_attn_use_vec_ಮಾರ್ಗ(ctx->webgpu_global_ctx, Q, K, V)) {
                    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_ನೇರ =
                        ggml_webgpu_flash_attn_k_ನೇರ(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ) ||
                        ggml_webgpu_flash_attn_v_ನೇರ(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ);
                    ಸ್ಥಿರಾಂಕ uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_ಗಾತ್ರ, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ, kv_ನೇರ);

                    ಸ್ಥಿರಾಂಕ uint32_t vec_nwg_cap = capabilities.min_subgroup_ಗಾತ್ರ;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ಸ್ಥಿರಾಂಕ size_t   align = capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;
                    ಸ್ಥಿರಾಂಕ uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    ಆದರೆ (nwg > 1u) {
                        ಸ್ಥಿರಾಂಕ uint64_t tmp_ದತ್ತಾಂಶ_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ಸ್ಥಿರಾಂಕ uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ಸ್ಥಿರಾಂಕ size_t   tmp_ಗಾತ್ರ_ಬೈಟುಗಳು = ROUNDUP_POW2((tmp_ದತ್ತಾಂಶ_elems + tmp_stats_elems) * ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_ಗಾತ್ರ_ಬೈಟುಗಳು + align;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    ಆದರೆ (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ) {
                        ಸ್ಥಿರಾಂಕ uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ಸ್ಥಿರಾಂಕ uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ಸ್ಥಿರಾಂಕ uint32_t stride_mask3    = (uint32_t) (ಮುಖವಾಡ->nb[3] / ggml_type_ಗಾತ್ರ(ಮುಖವಾಡ->type));
                        ಸ್ಥಿರಾಂಕ uint32_t blk_batch_ಎಣಿಕೆ = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ಸ್ಥಿರಾಂಕ uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_ಎಣಿಕೆ;
                        ಸ್ಥಿರಾಂಕ size_t   blk_ಗಾತ್ರ_ಬೈಟುಗಳು =
                            ROUNDUP_POW2(blk_elems * ಗಾತ್ರ(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_ಗಾತ್ರ_ಬೈಟುಗಳು + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = tensor->src[0];
                ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = tensor->src[1];
                ತಾರ್ಕಿಕ                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_ಗುಣಲಬ್ಧ,
                                             ctx->webgpu_global_ctx->vendor);
                ಆದರೆ (use_mmvq) {
                    ಸ್ಥಿರಾಂಕ size_t q8_src1_ಗಾತ್ರ = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_ಗಾತ್ರ +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = tensor->src[0];
                ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = tensor->src[1];
                ಆದರೆ (src0 && src1) {
                    ಸ್ಥಿರಾಂಕ size_t ids_cache_ಗಾತ್ರ = ಗಾತ್ರ(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    ಸ್ಥಿರಾಂಕ size_t ids_tiles_ಗಾತ್ರ = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_ಗಾತ್ರ + ids_tiles_ಗಾತ್ರ +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }
    ಹಿಂತಿರುಗಿ res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    ಹಿಂತಿರುಗಿ ctx->device_name.c_str();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    ಹಿಂತಿರುಗಿ ctx->device_desc.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_ಬಫರ್_ಗಾತ್ರ = ctx->webgpu_global_ctx->capabilities.limits.max_ಬಫರ್_ಗಾತ್ರ_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#ಆದರೆ UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_ಗಾತ್ರ = static_cast<uint64_t>(UINTPTR_MAX);
    ಆದರೆ (max_ಬಫರ್_ಗಾತ್ರ > max_ptr_ಗಾತ್ರ) {
        max_ಬಫರ್_ಗಾತ್ರ = max_ptr_ಗಾತ್ರ;
    }
#ಆದರೆ_ಅಂತ್ಯ
    *ಬಿಡುಗಡೆ  = static_cast<size_t>(max_ಬಫರ್_ಗಾತ್ರ);
    *ಒಟ್ಟು = static_cast<size_t>(max_ಬಫರ್_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                 = */ ಸುಳ್ಳು,
        /* .host_buffer           = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು,
        /* .events                = */ ಸುಳ್ಳು,
        /* .mmap_support          = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_guid_t ggml_backend_webgpu_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    ಹಿಂತಿರುಗಿ &guid;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_ಎಳೆಗಳು = ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup *
                         ctx->capabilities.limits.max_Compute_Workgroups_ಪ್ರತಿ_Dimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_ಬೈಟುಗಳು_ಪ್ರತಿ_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ, max_ಎಳೆಗಳು);
    std::vector<wgpu::Constant_ನಮೂದು> constants(2);
    constants[0].ಕೀಲಿ     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
    constants[1].ಕೀಲಿ     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#ಆದರೆ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_ಎಣಿಕೆ = 2;
    options.ಮುಂದಿನ_ಒಳ_Chain                   = &adapterTogglesDesc;
#ಆದರೆ_ಅಂತ್ಯ

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::Callback_ಬಹುಲಕ::AllowSpontaneous,
            [&adapter](wgpu::Request_Adapter_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2, wgpu::Adapter _adapter, wgpu::StringView ಸಂದೇಶ) {
                ಆದರೆ (ಸ್ಥಿತಿ_2 != wgpu::Request_Adapter_ಸ್ಥಿತಿ::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(ಸಂದೇಶ).c_str());
                    ಹಿಂತಿರುಗಿ;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

ಸ್ಥಿರ ಶೂನ್ಯ create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != ಶೂನ್ಯಸೂಚಕ);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#ಆದರೆ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::Adapter_Properties_Subgroup_ಮಾತೃಕೆ_Configs subgroup_ಮಾತೃಕೆ_configs{};
    ಆದರೆ (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ಮಾತೃಕೆ)) {
        info.ಮುಂದಿನ_ಒಳ_Chain = &subgroup_ಮಾತೃಕೆ_configs;
    }
#ಆದರೆ_ಅಂತ್ಯ
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_ಗಾತ್ರ = ggml_backend_webgpu_get_command_submit_batch_ಗಾತ್ರ();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_ಗುಣಲಬ್ಧ = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8_Integer_Dot_ಗುಣಲಬ್ಧ);

    ತಾರ್ಕಿಕ valid_subgroup_ಮಾತೃಕೆ_config = ಸುಳ್ಳು;
#ಆದರೆ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    ಆದರೆ (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ಮಾತೃಕೆ)) {
        ಗಾಗಿ (size_t i = 0; i < subgroup_ಮಾತೃಕೆ_configs.config_ಎಣಿಕೆ; i++) {
            ಸ್ಥಿರಾಂಕ wgpu::Subgroup_ಮಾತೃಕೆ_Config config = subgroup_ಮಾತೃಕೆ_configs.configs[i];
            ಆದರೆ (config.componentType == wgpu::Subgroup_ಮಾತೃಕೆ_Component_Type::F16 &&
                config.ಫಲಿತಾಂಶ_Component_Type == wgpu::Subgroup_ಮಾತೃಕೆ_Component_Type::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_ಮಾತೃಕೆ_config                  = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
        }
    }
#ಆದರೆ_ಅಂತ್ಯ
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_ಮಾತೃಕೆ = valid_subgroup_ಮಾತೃಕೆ_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_ಗಾತ್ರ = info.subgroup_Min_ಗಾತ್ರ;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_ಗಾತ್ರ = info.subgroup_Max_ಗಾತ್ರ;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#ಆದರೆ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    ಆದರೆ (ctx->webgpu_global_ctx->capabilities.supports_subgroup_ಮಾತೃಕೆ) {
        required_features.push_back(wgpu::FeatureName::Chromium_Experimental_Subgroup_ಮಾತೃಕೆ);
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#ಆದರೆ_ಅಂತ್ಯ

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.ದತ್ತಾಂಶ();
    dev_desc.required_Feature_ಎಣಿಕೆ = required_features.size();
    dev_desc.ಗಣ_Device_ಕಳೆದುಹೋದ_Callback(
        wgpu::Callback_ಬಹುಲಕ::AllowSpontaneous,
        [](ಸ್ಥಿರಾಂಕ wgpu::Device & device, wgpu::Device_ಕಳೆದುಹೋದ_Reason reason, wgpu::StringView ಸಂದೇಶ) {
            ಆದರೆ (reason == wgpu::Device_ಕಳೆದುಹೋದ_Reason::Destroyed) {
                ಹಿಂತಿರುಗಿ;
            }
            GGML_ಬಳಸದ(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<ಪೂರ್ಣಾಂಕ>(reason),
                           std::string(ಸಂದೇಶ).c_str());
        });
    dev_desc.ಗಣ_Uncaptured_Error_Callback(
        [](ಸ್ಥಿರಾಂಕ wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView ಸಂದೇಶ) {
            GGML_ಬಳಸದ(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<ಪೂರ್ಣಾಂಕ>(reason),
                       std::string(ಸಂದೇಶ).c_str());
        });

#ಆದರೆ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_ಎಣಿಕೆ  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_ಎಣಿಕೆ = 1;

    dev_desc.ಮುಂದಿನ_ಒಳ_Chain = &deviceTogglesDesc;
#ಆದರೆ_ಅಂತ್ಯ

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::Callback_ಬಹುಲಕ::AllowSpontaneous,
            [ctx](wgpu::Request_Device_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2, wgpu::Device device, wgpu::StringView ಸಂದೇಶ) {
                ಆದರೆ (ಸ್ಥಿತಿ_2 != wgpu::Request_Device_ಸ್ಥಿತಿ::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(ಸಂದೇಶ).c_str());
                    ಹಿಂತಿರುಗಿ;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != ಶೂನ್ಯಸೂಚಕ);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_ಬಫರ್(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು, wgpu::ಬಫರ್_Usage::CopyDst | wgpu::ಬಫರ್_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.Get_ಸರತಿ();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

ಸ್ಥಿರ webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು,
        webgpu_ctx->global_ctx->command_submit_batch_ಗಾತ್ರ + WEBGPU_ಅಂಕೆ_PARAM_ಸ್ಥಾನ_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_ಏಕರೂಪ_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
    ggml_webgpu_create_ಬಫರ್(webgpu_ctx->global_ctx->device, webgpu_ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_dev_error_buf,
                              WEBGPU_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ERROR_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು,
                              wgpu::ಬಫರ್_Usage::Storage | wgpu::ಬಫರ್_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_ಬಫರ್(webgpu_ctx->global_ctx->device, webgpu_ctx->ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_host_error_buf,
                              WEBGPU_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ERROR_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು,
                              wgpu::ಬಫರ್_Usage::CopyDst | wgpu::ಬಫರ್_Usage::MapRead, "set_rows_host_error_buf");

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = ಸುಳ್ಳು;
    ggml_webgpu_create_ಬಫರ್(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು,
        wgpu::ಬಫರ್_Usage::QueryResolve | wgpu::ಬಫರ್_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_ಬಫರ್(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_ಗಾತ್ರ_ಬೈಟುಗಳು,
                              wgpu::ಬಫರ್_Usage::CopyDst | wgpu::ಬಫರ್_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::Query_ಗಣ_Descriptor query_ಗಣ_desc = {};
    query_ಗಣ_desc.type                     = wgpu::QueryType::Timestamp;
    query_ಗಣ_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_ಎಣಿಕೆ;
    webgpu_ctx->profile_timestamp_query_ಗಣ = webgpu_ctx->global_ctx->device.Create_Query_ಗಣ(&query_ಗಣ_desc);
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_ಬಫರ್(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * ಗಾತ್ರ(uint32_t),
                              wgpu::ಬಫರ್_Usage::CopyDst | wgpu::ಬಫರ್_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_ಬಫರ್(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * ಗಾತ್ರ(uint32_t),
                              wgpu::ಬಫರ್_Usage::Storage | wgpu::ಬಫರ್_Usage::CopySrc, "debug_dev_buf");
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ webgpu_ctx;
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    GGML_ಬಳಸದ(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ಸ್ವಯಂಚಾಲಿತ * backend_ctx      = ಹೊಸ ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    ಸ್ವಯಂಚಾಲಿತ * backend = ಹೊಸ ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    ಹಿಂತಿರುಗಿ backend;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_webgpu_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_webgpu_ಬಫರ್_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_ಬಫರ್_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_ಬಫರ್_type_alloc_ಬಫರ್,
                        /* .alloc_buffer_n      = */ ಖಾಲಿ,
                        /* .get_alignment       = */ ggml_backend_webgpu_ಬಫರ್_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_ಬಫರ್_type_get_max_ಗಾತ್ರ,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
                        /* .get_alloc_size_n    = */ ಖಾಲಿ,
                        /* .is_host             = */ ಖಾಲಿ,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ ಖಾಲಿ
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_webgpu_ಬಫರ್_type;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ buft->iface.get_name == ggml_backend_webgpu_ಬಫರ್_type_get_name;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_webgpu_supported_qtype(ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    ಆದರೆ (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ ||
        (src0 != ಶೂನ್ಯಸೂಚಕ &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ) ||
        (src1 != ಶೂನ್ಯಸೂಚಕ &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ supports_op = ಸುಳ್ಳು;
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
            supports_op = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DUP:
        ಸ್ಥಿತಿ GGML_OP_CPY:
        ಸ್ಥಿತಿ GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ಆದರೆ (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            {
                ಆಯ್ಕೆ (src1->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        ಆಯ್ಕೆ (src0->type) {
                            ಸ್ಥಿತಿ GGML_TYPE_F32:
                            ಸ್ಥಿತಿ GGML_TYPE_F16:
                            ಸ್ಥಿತಿ GGML_TYPE_BF16:
                            ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                            ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                            ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                            ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                            ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                            ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                            ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                            ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                            ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                            ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                            ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                            ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                            ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                            ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                                supports_op = ನಿಜ;
                                ನಿಲ್ಲಿಸು;
                            ಪೂರ್ವನಿಯೋಜಿತ:
                                ನಿಲ್ಲಿಸು;
                        }
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ನಿಲ್ಲಿಸು;
                }
                ನಿಲ್ಲಿಸು;
            }
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            ಆಯ್ಕೆ (src1->type) {
                ಸ್ಥಿತಿ GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಆಯ್ಕೆ (src0->type) {
                        ಸ್ಥಿತಿ GGML_TYPE_F32:
                        ಸ್ಥಿತಿ GGML_TYPE_F16:
                        ಸ್ಥಿತಿ GGML_TYPE_BF16:
                        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                            supports_op = ನಿಜ;
                            ನಿಲ್ಲಿಸು;
                        ಪೂರ್ವನಿಯೋಜಿತ:
                            ನಿಲ್ಲಿಸು;
                    }
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                ಆದರೆ (!supports_op) {
                    ನಿಲ್ಲಿಸು;
                }
                ಆದರೆ (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ಸ್ಥಿರಾಂಕ size_t storage_ಸ್ಥಾನಾಂತರ_alignment = capabilities.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;

                // subgroup matrix path requirements
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_subgroup_ಮಾತೃಕೆ = ggml_webgpu_flash_attn_can_use_subgroup_ಮಾತೃಕೆ_ಮಾರ್ಗ(
                    capabilities.supports_subgroup_ಮಾತೃಕೆ, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_ಸ್ಥಾನಾಂತರ_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_ಸ್ಥಾನಾಂತರ_alignment));
                ಸ್ಥಿರಾಂಕ uint32_t k_tile_ತಲೆ_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ :
                                                       (uint32_t) ggml_blck_ಗಾತ್ರ(src1->type);
                ಸ್ಥಿರಾಂಕ uint32_t v_tile_ತಲೆ_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_ಅಗಲ :
                                                       (uint32_t) ggml_blck_ಗಾತ್ರ(src2->type);
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ     tile_kv_ತಲೆ_dims_aligned =
                    src0->ne[0] % k_tile_ತಲೆ_align == 0 && src2->ne[0] % v_tile_ತಲೆ_align == 0;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ tile_can_dispatch_all_q_ಅಡ್ಡಸಾಲುಗಳು =
                    capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_tile = !use_subgroup_ಮಾತೃಕೆ && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_ತಲೆ_dims_aligned && tile_can_dispatch_all_q_ಅಡ್ಡಸಾಲುಗಳು;

                ಆದರೆ (!use_subgroup_ಮಾತೃಕೆ && !use_tile) {
                    supports_op = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
                ಸ್ಥಿರಾಂಕ uint32_t q_tile =
                    use_subgroup_ಮಾತೃಕೆ ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ಸ್ಥಿರಾಂಕ uint32_t kv_granularity = use_subgroup_ಮಾತೃಕೆ ? capabilities.sg_mat_n : 1u;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ     kv_ನೇರ = use_subgroup_ಮಾತೃಕೆ ?
                                               ggml_webgpu_flash_attn_k_ನೇರ(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_ನೇರ(src0, src2, capabilities.sg_mat_k) :
                                               ಸುಳ್ಳು;
                ಸ್ಥಿರಾಂಕ uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_ಗಾತ್ರ, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != ಶೂನ್ಯಸೂಚಕ, kv_ನೇರ);
                supports_op = max_kv_tile > 0;
                ನಿಲ್ಲಿಸು;
            }
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(op)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ನಿಲ್ಲಿಸು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            {
                ಸ್ಥಿರಾಂಕ ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                ಆಯ್ಕೆ (UNARY_OP) {
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            {
                ಸ್ಥಿರಾಂಕ uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_ಪ್ರತಿ_Workgroup;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }
    ಆದರೆ (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ ||
        (src0 != ಶೂನ್ಯಸೂಚಕ &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ) ||
        (src1 != ಶೂನ್ಯಸೂಚಕ &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ) ||
        (src2 != ಶೂನ್ಯಸೂಚಕ &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_ಬಫರ್_Binding_ಗಾತ್ರ)) {
        supports_op = ಸುಳ್ಳು;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    ಆದರೆ (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } ಇಲ್ಲದಿದ್ದರೆ {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    ಹಿಂತಿರುಗಿ supports_op;
}

ಸ್ಥಿರ ರಚನೆ ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr = */ ಖಾಲಿ,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ ಖಾಲಿ,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    ಹಿಂತಿರುಗಿ ctx->name;
}

ಸ್ಥಿರ size_t ggml_backend_webgpu_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    ಹಿಂತಿರುಗಿ ctx->device_ಎಣಿಕೆ;
}

// Only one device is supported for now
ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ಸೂಚ್ಯಂಕ == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಆರಂಭ(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    ಸ್ಥಿರ ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    ಸ್ಥಿರ ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_ಒಟ್ಟು_ಕೊನೆ(reg_get_device, reg_ctx->webgpu_global_ctx);
    ಹಿಂತಿರುಗಿ &device;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ ಖಾಲಿ,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    ಸ್ಥಿರ ggml_backend_webgpu_reg_context * ctx = ಹೊಸ ggml_backend_webgpu_reg_context();

    ಸ್ಥಿರ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_ಎಣಿಕೆ = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    ಆದರೆ (ctx->webgpu_global_ctx != ಶೂನ್ಯಸೂಚಕ && ctx->webgpu_global_ctx->instance != ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.ದತ್ತಾಂಶ();
    instance_descriptor.required_Feature_ಎಣಿಕೆ                 = instance_features.size();

#ಆದರೆ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_ಎಣಿಕೆ = 1;
    instance_descriptor.ಮುಂದಿನ_ಒಳ_Chain        = &instanceTogglesDesc;
#ಆದರೆ_ಅಂತ್ಯ

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(ಹೊಸ webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    ಆದರೆ (ctx->webgpu_global_ctx->instance != ಶೂನ್ಯಸೂಚಕ) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    ಆದರೆ (adapter != ಶೂನ್ಯಸೂಚಕ && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#ಆದರೆ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#ಆದರೆ_ಅಂತ್ಯ
    ) {
        ctx->device_ಎಣಿಕೆ = 1;
    }

    ಹಿಂತಿರುಗಿ &reg;
}

ggml_backend_t ggml_backend_webgpu_init(ಶೂನ್ಯ) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    ಆದರೆ (ggml_backend_reg_dev_ಎಣಿಕೆ(reg) == 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ಹಿಂತಿರುಗಿ ggml_backend_webgpu_backend_init(dev, ಶೂನ್ಯಸೂಚಕ);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
