/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#అన్తర్భావ "ggml-webgpu.h"

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-webgpu-shader-lib.hpp"
#అన్తర్భావ "ggml.h"

#నిర్వచితమైతే __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#అయితే_ముగింపు

#అన్తర్భావ <webgpu/webgpu_cpp.h>

#అన్తర్భావ <atomic>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstring>
#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#అయితే_ముగింపు
#అన్తర్భావ <memory>
#అన్తర్భావ <mutex>
#అన్తర్భావ <optional>
#అన్తర్భావ <string>
#అన్తర్భావ <utility>
#అన్తర్భావ <vector>

#నిర్వచించు ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#నిర్వచించు CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
స్థిర అంతఃవరుస శూన్య compute_2d_workgroups(uint32_t మొత్తం_wg, uint32_t max_చొప్పున_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(మొత్తం_wg, max_చొప్పున_dim));
    wg_x = CEIL_DIV(మొత్తం_wg, wg_y);
}

స్థిర అంతఃవరుస uint32_t ggml_webgpu_u32_from_f32(దశాంశం value) {
    uint32_t బిట్లు;
    జ్ఞాపకం_నకలు(&బిట్లు, &value, పరిమాణం(బిట్లు));
    తిరిగి బిట్లు;
}

#నిర్వచితమైతే GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(సందేశం_2)  std::బయటకు << సందేశం_2 << std::పంక్తి_ముగింపు
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#లేకపోతే
#    define WEBGPU_LOG_DEBUG(సందేశం_2) ((శూన్య) 0)
#అయితే_ముగింపు  // GGML_WEBGPU_DEBUG

#నిర్వచితమైతే GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(id) స్వయంచాలక cpu_మొత్తం_ప్రారంభం##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_మొత్తం_చివర(id, ctx)                                                         \
        స్వయంచాలక   cpu_మొత్తం_చివర##id = std::chrono::high_resolution_clock::now();                            \
        ద్విగుణ cpu_మొత్తం_సమయం##id =                                                                      \
            std::chrono::duration<ద్విగుణ, std::milli>(cpu_మొత్తం_చివర##id - cpu_మొత్తం_ప్రారంభం##id).count(); \
        (ctx)->cpu_సమయం_ms[#id] += cpu_మొత్తం_సమయం##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_ప్రారంభం(id) స్వయంచాలక cpu_detail_ప్రారంభం##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_చివర(id, ctx)                                                          \
        స్వయంచాలక   cpu_detail_చివర##id = std::chrono::high_resolution_clock::now();                             \
        ద్విగుణ cpu_detail_సమయం##id =                                                                       \
            std::chrono::duration<ద్విగుణ, std::milli>(cpu_detail_చివర##id - cpu_detail_ప్రారంభం##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_సమయం##id;
#లేకపోతే
#    define WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(id)
#    define WEBGPU_CPU_PROFILE_మొత్తం_చివర(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_ప్రారంభం(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_చివర(id, ctx)
#అయితే_ముగింపు  // GGML_WEBGPU_CPU_PROFILE

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_లెక్క        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_కొలత_బైట్లు (WEBGPU_MAX_PROFILE_QUERY_లెక్క * పరిమాణం(uint64_t))
#అయితే_ముగింపు

/* Constants */

#నిర్వచించు WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_కొలత 64u
#నిర్వచించు WEBGPU_అంకె_PARAM_స్థానం_SAFETY_MARGIN      10u
#నిర్వచించు WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#నిర్వచించు WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#నిర్వచించు WEBGPU_PARAMS_BUF_కొలత_బైట్లు             128  // enough for 32 parameters
#నిర్వచించు WEBGPU_సమితి_అడ్డువరుసలు_ERROR_BUF_కొలత_బైట్లు     4
#నిర్వచించు WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
స్థిర శూన్య * స్థిరరాశి webgpu_ptr_పునాది = (శూన్య *) (uintptr_t) 0x1000;  // NOLINT

స్థిర size_t ggml_webgpu_tensor_స్థానభ్రంశం(స్థిరరాశి ggml_tensor * tensor) {
    స్థిరరాశి ggml_tensor * పునాది_tensor = tensor->view_src ? tensor->view_src : tensor;
    తిరిగి (size_t) ((uintptr_t) పునాది_tensor->దత్తాంశం - (uintptr_t) webgpu_ptr_పునాది) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
స్థిర శూన్య ggml_webgpu_create_బఫరు(wgpu::Device &    device,
                                      wgpu::బఫరు_2 &    బఫరు,
                                      size_t            size,
                                      wgpu::బఫరు_Usage usage,
                                      స్థిరరాశి అక్షరం *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
నిర్మాణం webgpu_param_arena {
    wgpu::బఫరు_2 బఫరు;
    size_t       స్థానం_stride = 0;
    size_t       స్థానం_కొలత   = 0;
    uint32_t     స్థానం_లెక్క  = 0;
    uint32_t     తదుపరి_స్థానం   = 0;

    శూన్య init(wgpu::Device device, size_t స్థానం_కొలత, uint32_t స్థానం_లెక్క, size_t alignment) {
        ఇది->స్థానం_stride = ROUNDUP_POW2(స్థానం_కొలత, alignment);
        ఇది->స్థానం_కొలత   = స్థానం_కొలత;
        ఇది->స్థానం_లెక్క  = స్థానం_లెక్క;
        ఇది->తదుపరి_స్థానం   = 0;

        ggml_webgpu_create_బఫరు(device, బఫరు, ఇది->స్థానం_stride * స్థానం_లెక్క,
                                  wgpu::బఫరు_Usage::CopyDst | wgpu::బఫరు_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_స్థానం(size_t size) {
        GGML_ASSERT(size <= స్థానం_కొలత);
        ఒకవేళ (తదుపరి_స్థానం >= స్థానం_లెక్క) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        తిరిగి స్థానం_stride * తదుపరి_స్థానం++;
    }

    శూన్య reset() { తదుపరి_స్థానం = 0; }

    శూన్య cleanup() {
        ఒకవేళ (బఫరు) {
            బఫరు.Destroy();
            బఫరు = శూన్యసూచిక;
        }
    }

    ~webgpu_param_arena() { ఇది->cleanup(); }
};

నిర్మాణం webgpu_encoded_op {
    uint32_t అంకె_kernels = 0;
#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_పేర్లు;
#అయితే_ముగింపు
};

నిర్మాణం webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_సమూహం_నమోదు> bind_సమూహం_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

నిర్మాణం webgpu_capabilities {
    wgpu::Limits limits;
    తార్కిక         supports_subgroups       = అబద్ధం;
    తార్కిక         supports_subgroup_మాతృక = అబద్ధం;
    తార్కిక         supports_dot_లబ్ధం     = అబద్ధం;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_కొలత     = 0;
    uint32_t min_subgroup_కొలత = 0;
    uint32_t max_subgroup_కొలత = 0;
    size_t   memset_బైట్లు_చొప్పున_thread;
};

// Stores global webgpu members
నిర్మాణం webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::వరుస_2    queue;
    uint32_t       command_submit_batch_కొలత = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_కొలత;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::బఫరు_2         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::బఫరు_2    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#నిర్వచితమైతే GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, ద్విగుణ> cpu_సమయం_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, ద్విగుణ> cpu_detail_ms;
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_WEBGPU_DEBUG
    wgpu::బఫరు_2 debug_host_buf;
    wgpu::బఫరు_2 debug_dev_buf;
#అయితే_ముగింపు

    ~webgpu_global_context_struct() {
        ఒకవేళ (ఇది->get_tensor_staging_buf) {
            ఇది->get_tensor_staging_buf.Destroy();
            ఇది->get_tensor_staging_buf = శూన్యసూచిక;
        }
        ఒకవేళ (ఇది->memset_params_buf) {
            ఇది->memset_params_buf.Destroy();
            ఇది->memset_params_buf = శూన్యసూచిక;
        }
#నిర్వచితమైతే GGML_WEBGPU_DEBUG
        ఒకవేళ (ఇది->debug_host_buf) {
            ఇది->debug_host_buf.Destroy();
            ఇది->debug_host_buf = శూన్యసూచిక;
        }
        ఒకవేళ (ఇది->debug_dev_buf) {
            ఇది->debug_dev_buf.Destroy();
            ఇది->debug_dev_buf = శూన్యసూచిక;
        }
#అయితే_ముగింపు
    }
};

రకనిర్వచనం std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
నిర్మాణం webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::బఫరు_2             సమితి_అడ్డువరుసలు_dev_error_buf;
    wgpu::బఫరు_2             సమితి_అడ్డువరుసలు_host_error_buf;
    wgpu::CommandEncoder     చురుకైన_command_encoder;
    wgpu::ComputePassEncoder చురుకైన_compute_pass;
    తార్కిక                     batch_compute_passes = నిజం;

    size_t memset_బైట్లు_చొప్పున_thread;

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, ద్విగుణ> shader_gpu_సమయం_ms;
    wgpu::బఫరు_2                            profile_timestamp_dev_buf;
    wgpu::బఫరు_2                            profile_timestamp_host_buf;
    wgpu::Query_సమితి                          profile_timestamp_query_సమితి;
    uint32_t                                profile_timestamp_query_లెక్క = 0;
#అయితే_ముగింపు

    ~webgpu_context_struct() {
#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
        ఒకవేళ (ఇది->profile_timestamp_host_buf) {
            ఇది->profile_timestamp_host_buf.Destroy();
            ఇది->profile_timestamp_host_buf = శూన్యసూచిక;
        }
        ఒకవేళ (ఇది->profile_timestamp_dev_buf) {
            ఇది->profile_timestamp_dev_buf.Destroy();
            ఇది->profile_timestamp_dev_buf = శూన్యసూచిక;
        }
        ఒకవేళ (ఇది->profile_timestamp_query_సమితి) {
            ఇది->profile_timestamp_query_సమితి.Destroy();
            ఇది->profile_timestamp_query_సమితి = శూన్యసూచిక;
        }
#అయితే_ముగింపు
        ఒకవేళ (ఇది->సమితి_అడ్డువరుసలు_host_error_buf) {
            ఇది->సమితి_అడ్డువరుసలు_host_error_buf.Destroy();
            ఇది->సమితి_అడ్డువరుసలు_host_error_buf = శూన్యసూచిక;
        }
        ఒకవేళ (ఇది->సమితి_అడ్డువరుసలు_dev_error_buf) {
            ఇది->సమితి_అడ్డువరుసలు_dev_error_buf.Destroy();
            ఇది->సమితి_అడ్డువరుసలు_dev_error_buf = శూన్యసూచిక;
        }
    }
};

రకనిర్వచనం std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
నిర్మాణం ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_లెక్క;
    స్థిరరాశి అక్షరం *          name;
};

// Per-device struct for the global logical device interface
నిర్మాణం ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
నిర్మాణం ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
నిర్మాణం ggml_backend_webgpu_బఫరు_context {
    wgpu::బఫరు_2          బఫరు;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_బఫరు_context(wgpu::బఫరు_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        బఫరు(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

స్థిర webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   స్థిరరాశి అక్షరం *                             shader_సంకేతం,
                                                   స్థిరరాశి అక్షరం *                             label,
                                                   స్థిరరాశి std::vector<wgpu::Constant_నమోదు> & constants = {}) {
    wgpu::Shader_ఆధారం_WGSL shader_ఆధారం;
    shader_ఆధారం.code = shader_సంకేతం;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.తదుపరి_లోపల_Chain = &shader_ఆధారం;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.నమోదు_బిందువు_2 = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = శూన్యసూచిక;  // nullptr means auto layout
    ఒకవేళ (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.దత్తాంశం();
        pipeline_desc.compute.constant_లెక్క = constants.size();
    }
    తిరిగి { device.CreateComputePipeline(&pipeline_desc), label };
}

స్థిర శూన్య ggml_webgpu_create_బఫరు(wgpu::Device &    device,
                                      wgpu::బఫరు_2 &    బఫరు,
                                      size_t            size,
                                      wgpu::బఫరు_Usage usage,
                                      స్థిరరాశి అక్షరం *      label) {
    wgpu::బఫరు_Descriptor బఫరు_desc;
    బఫరు_desc.size             = size;
    బఫరు_desc.usage            = usage;
    బఫరు_desc.label            = label;
    బఫరు_desc.mappedAtCreation = అబద్ధం;

    // TODO: error handling
    బఫరు = device.Create_బఫరు(&బఫరు_desc);
}

స్థిర wgpu::బఫరు_2 ggml_webgpu_tensor_buf(స్థిరరాశి ggml_tensor * tensor) {
    ggml_backend_webgpu_బఫరు_context * ctx = (ggml_backend_webgpu_బఫరు_context *) tensor->బఫరు->context;
    తిరిగి ctx->బఫరు;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
స్థిర size_t ggml_webgpu_tensor_align_స్థానభ్రంశం(స్థిరరాశి ggml_tensor * t, size_t alignment) {
    స్థిరరాశి size_t స్థానభ్రంశం    = ggml_webgpu_tensor_స్థానభ్రంశం(t);
    స్థిరరాశి size_t type_కొలత = ggml_type_కొలత(t->type);
    size_t       aligned   = స్థానభ్రంశం & ~(alignment - 1);
    అయితే ((స్థానభ్రంశం - aligned) % type_కొలత != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    తిరిగి aligned;
}

స్థిర size_t ggml_webgpu_tensor_misalignment(స్థిరరాశి ggml_tensor * t, size_t alignment) {
    తిరిగి ggml_webgpu_tensor_స్థానభ్రంశం(t) - ggml_webgpu_tensor_align_స్థానభ్రంశం(t, alignment);
}

స్థిర size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, స్థిరరాశి ggml_tensor * t) {
    తిరిగి ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
}

స్థిర size_t ggml_webgpu_tensor_align_స్థానభ్రంశం(webgpu_context & ctx, స్థిరరాశి ggml_tensor * t) {
    తిరిగి ggml_webgpu_tensor_align_స్థానభ్రంశం(t, ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
}

స్థిర size_t ggml_webgpu_tensor_binding_కొలత(స్థిరరాశి ggml_tensor * t, size_t alignment) {
    తిరిగి ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

స్థిర size_t ggml_webgpu_tensor_binding_కొలత(webgpu_context & ctx, స్థిరరాశి ggml_tensor * t) {
    తిరిగి ggml_webgpu_tensor_binding_కొలత(t, ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
}

స్థిర తార్కిక ggml_webgpu_tensor_binding_overlap(స్థిరరాశి webgpu_global_context & global_ctx,
                                               స్థిరరాశి ggml_tensor *           a,
                                               స్థిరరాశి ggml_tensor *           b) {
    ఒకవేళ (a->బఫరు != b->బఫరు) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి size_t alignment = global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;
    స్థిరరాశి size_t a_స్థానభ్రంశం  = ggml_webgpu_tensor_align_స్థానభ్రంశం(a, alignment);
    స్థిరరాశి size_t b_స్థానభ్రంశం  = ggml_webgpu_tensor_align_స్థానభ్రంశం(b, alignment);
    తిరిగి a_స్థానభ్రంశం < b_స్థానభ్రంశం + ggml_webgpu_tensor_binding_కొలత(b, alignment) &&
           b_స్థానభ్రంశం < a_స్థానభ్రంశం + ggml_webgpu_tensor_binding_కొలత(a, alignment);
}

స్థిర తార్కిక ggml_webgpu_tensor_binding_overlap_range(స్థిరరాశి webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_బఫరు_t         బఫరు,
                                                     size_t                        స్థానభ్రంశం,
                                                     size_t                        size) {
    ఒకవేళ (tensor->బఫరు != బఫరు) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి size_t alignment     = global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;
    స్థిరరాశి size_t tensor_స్థానభ్రంశం = ggml_webgpu_tensor_align_స్థానభ్రంశం(tensor, alignment);
    తిరిగి tensor_స్థానభ్రంశం < స్థానభ్రంశం + size && స్థానభ్రంశం < tensor_స్థానభ్రంశం + ggml_webgpu_tensor_binding_కొలత(tensor, alignment);
}

నిర్మాణం ggml_webgpu_merged_binding_range {
    size_t స్థానభ్రంశం;
    size_t size;
};

స్థిర ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_స్థానభ్రంశం = SIZE_MAX;
    size_t merged_చివర    = 0;

    కొరకు (ggml_tensor * tensor : tensors) {
        స్థిరరాశి size_t bind_స్థానభ్రంశం = ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, tensor);
        స్థిరరాశి size_t bind_చివర    = bind_స్థానభ్రంశం + ggml_webgpu_tensor_binding_కొలత(ctx, tensor);

        merged_స్థానభ్రంశం = std::min(merged_స్థానభ్రంశం, bind_స్థానభ్రంశం);
        merged_చివర    = std::max(merged_చివర, bind_చివర);
    }

    తిరిగి { merged_స్థానభ్రంశం, merged_చివర - merged_స్థానభ్రంశం };
}

స్థిర uint32_t ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(స్థిరరాశి ggml_tensor *                      tensor,
                                                         స్థిరరాశి ggml_webgpu_merged_binding_range & merged_range) {
    తిరిగి (uint32_t) ((ggml_webgpu_tensor_స్థానభ్రంశం(tensor) - merged_range.స్థానభ్రంశం) / ggml_type_కొలత(tensor->type));
}

స్థిర wgpu::Bind_సమూహం_నమోదు ggml_webgpu_make_bind_సమూహం_నమోదు(uint32_t     binding,
                                                              wgpu::బఫరు_2 బఫరు,
                                                              uint64_t     స్థానభ్రంశం,
                                                              uint64_t     size) {
    wgpu::Bind_సమూహం_నమోదు నమోదు_2 = {};
    నమోదు_2.binding              = binding;
    నమోదు_2.బఫరు               = std::move(బఫరు);
    నమోదు_2.స్థానభ్రంశం               = స్థానభ్రంశం;
    నమోదు_2.size                 = size;
    తిరిగి నమోదు_2;
}

స్థిర wgpu::Bind_సమూహం_నమోదు ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    తిరిగి ggml_webgpu_make_bind_సమూహం_నమోదు(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, tensor),
                                             ggml_webgpu_tensor_binding_కొలత(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

మూస <రకం_పేరు T>
స్థిర శూన్య ggml_backend_webgpu_check_wait_స్థితి(wgpu::Wait_స్థితి wait_స్థితి,
                                                  T                callback_స్థితి,
                                                  T                success_స్థితి,
                                                  స్థిరరాశి అక్షరం *     wait_name,
                                                  స్థిరరాశి అక్షరం *     failure_name,
                                                  స్థిరరాశి అక్షరం *     callback_సందేశం) {
    ఒకవేళ (wait_స్థితి == wgpu::Wait_స్థితి::Timed_బహిర్గతం) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    ఒకవేళ (wait_స్థితి == wgpu::Wait_స్థితి::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    ఒకవేళ (callback_స్థితి != success_స్థితి) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<పూర్ణాంకం>(callback_స్థితి),
                   callback_సందేశం);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
స్థిర uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    తిరిగి UINT32_MAX;
}

స్థిర uint32_t ggml_backend_webgpu_get_command_submit_batch_కొలత() {
    తిరిగి WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_కొలత;
}

స్థిర శూన్య ggml_backend_webgpu_wait_వరుస(webgpu_global_context & ctx) {
    wgpu::వరుస_Work_పూర్తి_స్థితి callback_స్థితి = wgpu::వరుస_Work_పూర్తి_స్థితి::Error;
    std::string               callback_సందేశం;

    స్థిరరాశి wgpu::Wait_స్థితి wait_స్థితి = ctx->instance.WaitAny(
        ctx->queue.On_Submitted_Work_పూర్తి(
            wgpu::Callback_బహుళకం::AllowSpontaneous,
            [&callback_స్థితి, &callback_సందేశం](wgpu::వరుస_Work_పూర్తి_స్థితి స్థితి_2, wgpu::StringView సందేశం) {
                callback_స్థితి  = స్థితి_2;
                callback_సందేశం = std::string(సందేశం);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_స్థితి(wait_స్థితి, callback_స్థితి, wgpu::వరుస_Work_పూర్తి_స్థితి::Success,
                                          "Queue wait", "Queue work", callback_సందేశం.c_str());
}

స్థిర శూన్య ggml_backend_webgpu_map_బఫరు(webgpu_global_context & ctx,
                                           wgpu::బఫరు_2 &          బఫరు,
                                           wgpu::Map_బహుళకం           mode,
                                           size_t                  స్థానభ్రంశం,
                                           size_t                  size) {
    wgpu::Map_Async_స్థితి callback_స్థితి = wgpu::Map_Async_స్థితి::Error;
    std::string          callback_సందేశం;

    స్థిరరాశి wgpu::Wait_స్థితి wait_స్థితి = ctx->instance.WaitAny(
        బఫరు.MapAsync(mode, స్థానభ్రంశం, size, wgpu::Callback_బహుళకం::AllowSpontaneous,
                        [&callback_స్థితి, &callback_సందేశం](wgpu::Map_Async_స్థితి స్థితి_2, wgpu::StringView సందేశం) {
                            callback_స్థితి  = స్థితి_2;
                            callback_సందేశం = std::string(సందేశం);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_స్థితి(wait_స్థితి, callback_స్థితి, wgpu::Map_Async_స్థితి::Success,
                                          "Buffer map wait", "Buffer map", callback_సందేశం.c_str());
}

స్థిర శూన్య ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                స్థిరరాశి wgpu::Command_బఫరు commands,
                                                uint32_t &                అంకె_inflight_batches) {
    ఒకవేళ (అంకె_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_వరుస(ctx->global_ctx);
        అంకె_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    అంకె_inflight_batches++;
}

#నిర్వచితమైతే GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
స్థిర శూన్య ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_బఫరు_గమ్యం_బఫరు(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_కొలత());
    wgpu::Command_బఫరు commands = encoder.ముగింపు_2();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_బఫరు(ctx, ctx->debug_host_buf, wgpu::Map_బహుళకం::Read, 0, ctx->debug_host_buf.Get_కొలత());
    స్థిరరాశి దశాంశం * debug_దత్తాంశం = (స్థిరరాశి దశాంశం *) ctx->debug_host_buf.GetConstMappedRange();
    std::బయటకు << "debug[0]: " << debug_దత్తాంశం[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#అయితే_ముగింపు

స్థిర webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         స్థిరరాశి std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            ఫలితం = {};
    std::vector<wgpu::Bind_సమూహం> bind_groups;
    std::vector<size_t>          param_offsets;
    ఫలితం.అంకె_kernels = dispatches.size();

    కొరకు (size_t i = 0; i < dispatches.size(); i++) {
        స్థిరరాశి webgpu_dispatch_desc & dispatch     = dispatches[i];
        స్థిరరాశి size_t                 param_కొలత   = dispatch.params.size() * పరిమాణం(uint32_t);
        స్థిరరాశి size_t                 param_స్థానభ్రంశం = ctx->param_arena.alloc_స్థానం(param_కొలత);

        std::vector<wgpu::Bind_సమూహం_నమోదు> entries            = dispatch.bind_సమూహం_entries;
        uint32_t                          params_binding_అంకె = entries.size();
        entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(params_binding_అంకె, ctx->param_arena.బఫరు, param_స్థానభ్రంశం,
                                                            ctx->param_arena.స్థానం_కొలత));

        wgpu::Bind_సమూహం_Descriptor bind_సమూహం_desc;
        bind_సమూహం_desc.layout     = dispatch.pipeline.pipeline.Get_Bind_సమూహం_Layout(0);
        bind_సమూహం_desc.నమోదు_లెక్క = entries.size();
        bind_సమూహం_desc.entries    = entries.దత్తాంశం();
        bind_సమూహం_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.Create_Bind_సమూహం(&bind_సమూహం_desc));
        param_offsets.push_back(param_స్థానభ్రంశం);
    }

    కొరకు (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_బఫరు(ctx->param_arena.బఫరు, param_offsets[i], dispatches[i].params.దత్తాంశం(),
                                           dispatches[i].params.size() * పరిమాణం(uint32_t));
    }

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    కొరకు (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_లెక్క + 2 <= WEBGPU_MAX_PROFILE_QUERY_లెక్క);
        స్థిరరాశి uint32_t query_begin = ctx->profile_timestamp_query_లెక్క++;
        స్థిరరాశి uint32_t query_చివర   = ctx->profile_timestamp_query_లెక్క++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.query_సమితి                    = ctx->profile_timestamp_query_సమితి;
        ts_writes.beginning_యొక్క_Pass_Write_సూచీ   = query_begin;
        ts_writes.చివర_యొక్క_Pass_Write_సూచీ         = query_చివర;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->చురుకైన_command_encoder.BeginComputePass(&pass_desc);

        pass.సమితి_Pipeline(dispatches[i].pipeline.pipeline);
        pass.సమితి_Bind_సమూహం(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.మొదటి, dispatches[i].workgroups.second, 1);
        pass.చివర_2();
        ఫలితం.pipeline_పేర్లు.push_back(dispatches[i].pipeline.name);
    }
#లేకపోతే
    కొరకు (size_t i = 0; i < dispatches.size(); i++) {
        ఒకవేళ (ctx->batch_compute_passes) {
            ctx->చురుకైన_compute_pass.సమితి_Pipeline(dispatches[i].pipeline.pipeline);
            ctx->చురుకైన_compute_pass.సమితి_Bind_సమూహం(0, bind_groups[i]);
            ctx->చురుకైన_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.మొదటి, dispatches[i].workgroups.second,
                                                        1);
        } లేకపోతే {
            wgpu::ComputePassEncoder pass = ctx->చురుకైన_command_encoder.BeginComputePass();
            pass.సమితి_Pipeline(dispatches[i].pipeline.pipeline);
            pass.సమితి_Bind_సమూహం(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.మొదటి, dispatches[i].workgroups.second, 1);
            pass.చివర_2();
        }
    }
#అయితే_ముగింపు

    తిరిగి ఫలితం;
}

స్థిర webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::Bind_సమూహం_నమోదు> bind_సమూహం_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    తిరిగి ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_సమూహం_entries), { wg_x, wg_y } },
    });
}

స్థిర శూన్య ggml_backend_webgpu_బఫరు_memset(webgpu_global_context & ctx,
                                              wgpu::బఫరు_2 &          buf,
                                              uint32_t                value,
                                              size_t                  స్థానభ్రంశం,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) స్థానభ్రంశం, (uint32_t) size, value };
    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_bind_సమూహం_నమోదు(0, buf, 0, buf.Get_కొలత()) };
    size_t                            బైట్లు_చొప్పున_wg =
        ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup * ctx->capabilities.memset_బైట్లు_చొప్పున_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, బైట్లు_చొప్పున_wg);

    ctx->queue.Write_బఫరు(ctx->memset_params_buf, 0, params.దత్తాంశం(), params.size() * పరిమాణం(uint32_t));

    wgpu::Bind_సమూహం_నమోదు params_నమోదు = {};
    params_నమోదు.binding              = 1;
    params_నమోదు.బఫరు               = ctx->memset_params_buf;
    params_నమోదు.స్థానభ్రంశం               = 0;
    params_నమోదు.size                 = WEBGPU_PARAMS_BUF_కొలత_బైట్లు;
    entries.push_back(params_నమోదు);

    wgpu::Bind_సమూహం_Descriptor bind_సమూహం_desc;
    bind_సమూహం_desc.layout     = ctx->memset_pipeline.pipeline.Get_Bind_సమూహం_Layout(0);
    bind_సమూహం_desc.నమోదు_లెక్క = entries.size();
    bind_సమూహం_desc.entries    = entries.దత్తాంశం();
    bind_సమూహం_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::Bind_సమూహం bind_సమూహం = ctx->device.Create_Bind_సమూహం(&bind_సమూహం_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.సమితి_Pipeline(ctx->memset_pipeline.pipeline);
    pass.సమితి_Bind_సమూహం(0, bind_సమూహం);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.చివర_2();

    wgpu::Command_బఫరు              command  = encoder.ముగింపు_2();
    std::vector<wgpu::Command_బఫరు> commands = { command };
    ctx->queue.Submit(commands.size(), commands.దత్తాంశం());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

స్థిర స్థిరరాశి అక్షరం * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    తిరిగి ctx->name.c_str();
}

స్థిర శూన్య ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#నిర్వచితమైతే GGML_WEBGPU_CPU_PROFILE
    std::బయటకు << "\n[ggml_webgpu cpu profiling summary]\n";
    ద్విగుణ మొత్తం_cpu = 0.0;
    కొరకు (స్థిరరాశి స్వయంచాలక & kv : ctx->webgpu_ctx->global_ctx->cpu_సమయం_ms) {
        మొత్తం_cpu += kv.second;
    }
    std::బయటకు << "ggml_webgpu: total cpu time: " << మొత్తం_cpu << " ms\n";
    std::బయటకు << "ggml_webgpu: cpu breakdown:\n";
    కొరకు (స్థిరరాశి స్వయంచాలక & kv : ctx->webgpu_ctx->global_ctx->cpu_సమయం_ms) {
        ద్విగుణ pct = (మొత్తం_cpu > 0.0) ? (kv.second / మొత్తం_cpu * 100.0) : 0.0;
        std::బయటకు << "ggml_webgpu:  " << kv.మొదటి << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    ఒకవేళ (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::బయటకు << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        ద్విగుణ pct = (మొత్తం_cpu > 0.0) ? (kv.second / మొత్తం_cpu * 100.0) : 0.0;
        std::బయటకు << "ggml_webgpu:  " << kv.మొదటి << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    std::బయటకు << "\n[ggml_webgpu gpu profiling summary]\n";
    ద్విగుణ మొత్తం_gpu = 0.0;
    కొరకు (స్థిరరాశి స్వయంచాలక & kv : ctx->webgpu_ctx->shader_gpu_సమయం_ms) {
        మొత్తం_gpu += kv.second;
    }
    std::బయటకు << "ggml_webgpu: total gpu time (all shaders): " << మొత్తం_gpu << " ms\n";
    std::బయటకు << "\nggml_webgpu: gpu breakdown:\n";
    కొరకు (స్థిరరాశి స్వయంచాలక & kv : ctx->webgpu_ctx->shader_gpu_సమయం_ms) {
        ద్విగుణ pct = (మొత్తం_gpu > 0.0) ? (kv.second / మొత్తం_gpu * 100.0) : 0.0;
        std::బయటకు << "ggml_webgpu:  " << kv.మొదటి << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::బయటకు << "ggml_webgpu: gpu/cpu ratio: " << (మొత్తం_cpu > 0.0 ? మొత్తం_gpu / మొత్తం_cpu : 0.0) << "\n";
#అయితే_ముగింపు

    తొలగించు ctx;
    తొలగించు backend;
}

స్థిర webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_కొలత(src->type)), (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)), (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_కొలత(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ne, decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_సమితి(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_సమితి_pipeline(shader_lib_ctx);

    స్వయంచాలక *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    స్థిరరాశి తార్కిక inplace   = decisions->inplace;

    స్థిరరాశి uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    స్థిరరాశి uint32_t dst_type_కొలత = (uint32_t) ggml_type_కొలత(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (((స్థిరరాశి int32_t *) dst->op_params)[3] / dst_type_కొలత),

        (uint32_t) (src1->nb[0] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),

        1u,
        (uint32_t) (((స్థిరరాశి int32_t *) dst->op_params)[0] / dst_type_కొలత),
        (uint32_t) (((స్థిరరాశి int32_t *) dst->op_params)[1] / dst_type_కొలత),
        (uint32_t) (((స్థిరరాశి int32_t *) dst->op_params)[2] / dst_type_కొలత),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries;
    uint32_t                          binding_సూచీ = 0;
    ఒకవేళ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0));
        binding_సూచీ++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_సూచీ, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_సూచీ + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ne, decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    స్థిరరాశి uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)),
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

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ne, decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
    shader_lib_ctx.wg_mem_హద్దు_బైట్లు = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_కొలత;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst),
    };

    స్థిరరాశి uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_కొలత);
    స్థిరరాశి uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    స్థిరరాశి int32_t s0 = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి int32_t s1 = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి int32_t p0 = ggml_get_op_params_i32(dst, 2);
    స్థిరరాశి int32_t p1 = ggml_get_op_params_i32(dst, 3);
    స్థిరరాశి int32_t d0 = ggml_get_op_params_i32(dst, 4);
    స్థిరరాశి int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),

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

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
స్థిర webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    స్థిరరాశి int32_t s0 = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి int32_t s1 = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి int32_t p0 = ggml_get_op_params_i32(dst, 2);
    స్థిరరాశి int32_t p1 = ggml_get_op_params_i32(dst, 3);
    స్థిరరాశి int32_t d0 = ggml_get_op_params_i32(dst, 4);
    స్థిరరాశి int32_t d1 = ggml_get_op_params_i32(dst, 5);

    // Scalar params matching conv2d_dw.wgsl (weight src0 [KW,KH,1,C], input src1, output dst).
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),

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

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    స్థిరరాశి తార్కిక      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    స్థిరరాశి int32_t s0    = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి int32_t s1    = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి int32_t p0    = ggml_get_op_params_i32(dst, 2);
    స్థిరరాశి int32_t p1    = ggml_get_op_params_i32(dst, 3);
    స్థిరరాశి int32_t d0    = ggml_get_op_params_i32(dst, 4);
    స్థిరరాశి int32_t d1    = ggml_get_op_params_i32(dst, 5);
    స్థిరరాశి తార్కిక    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    స్థిరరాశి uint32_t KW = src0->ne[0];
    స్థిరరాశి uint32_t KH = is_2D ? src0->ne[1] : 1;
    స్థిరరాశి uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    స్థిరరాశి uint32_t IW = src1->ne[0];
    స్థిరరాశి uint32_t IH = is_2D ? src1->ne[1] : 1;
    స్థిరరాశి uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    స్థిరరాశి uint32_t OW = dst->ne[1];
    స్థిరరాశి uint32_t OH = is_2D ? dst->ne[2] : 1;

    స్థిరరాశి uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_కొలత(src1->type));
    స్థిరరాశి uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)) : 0;
    స్థిరరాశి uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type));
    స్థిరరాశి uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type));

    స్థిరరాశి uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_కొలత(dst->type));
    స్థిరరాశి uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type));
    స్థిరరాశి uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)) : 0;
    స్థిరరాశి uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),

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

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    స్థిరరాశి uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->టోకెన్లు_చొప్పున_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst),
    };

    స్థిరరాశి uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->ఖండం_కొలత_3);
    స్థిరరాశి uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_ssm_పరిశీలన(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_కొలత        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    ఒకవేళ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    ఒకవేళ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    ఒకవేళ (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_పరిశీలన_pipeline(shader_lib_ctx, xbc_overlap);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_ssm_పరిశీలన_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t స్థానభ్రంశం_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type));
    uint32_t స్థానభ్రంశం_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_కొలత(src2->type));
    uint32_t స్థానభ్రంశం_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_కొలత(src3->type));
    uint32_t స్థానభ్రంశం_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_కొలత(src4->type));
    uint32_t స్థానభ్రంశం_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_కొలత(src5->type));
    uint32_t స్థానభ్రంశం_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_కొలత(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    ఒకవేళ (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        స్థానభ్రంశం_x         = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src1, xbc_merged_range);
        స్థానభ్రంశం_B         = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src4, xbc_merged_range);
    } లేకపోతే ఒకవేళ (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        స్థానభ్రంశం_B         = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src4, xbc_merged_range);
        స్థానభ్రంశం_C         = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src5, xbc_merged_range);
    } లేకపోతే ఒకవేళ (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        స్థానభ్రంశం_x         = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src1, xbc_merged_range);
        స్థానభ్రంశం_B         = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src4, xbc_merged_range);
        స్థానభ్రంశం_C         = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_స్థానభ్రంశం = 0;
    size_t xbc_bind_కొలత   = 0;
    ఒకవేళ (xbc_overlap > 0) {
        xbc_bind_స్థానభ్రంశం = xbc_merged_range.స్థానభ్రంశం;
        xbc_bind_కొలత   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        స్థానభ్రంశం_x,
        స్థానభ్రంశం_dt,
        స్థానభ్రంశం_A,
        స్థానభ్రంశం_B,
        స్థానభ్రంశం_C,
        స్థానభ్రంశం_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_కొలత(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_కొలత(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_కొలత(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_కొలత(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_కొలత(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_కొలత(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_కొలత(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_కొలత(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_కొలత(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_అంకె = 0;

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, src0),
    };
    // xbc_merged binding
    ఒకవేళ (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(binding_అంకె++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_స్థానభ్రంశం, xbc_bind_కొలత));
    }
    // x
    ఒకవేళ (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, src1));
    }
    // B
    ఒకవేళ (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, src4));
    }
    // C
    ఒకవేళ (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె++, dst));

    స్థిరరాశి uint32_t మొత్తం_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    స్థిరరాశి uint32_t max_wg_చొప్పున_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(మొత్తం_wg, max_wg_చొప్పున_dim, wg_x, wg_y);

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    స్థిరరాశి uint32_t s_v      = (uint32_t) src2->ne[0];
    స్థిరరాశి uint32_t h        = (uint32_t) src2->ne[1];
    స్థిరరాశి uint32_t n_టోకెన్లు = (uint32_t) src2->ne[2];
    స్థిరరాశి uint32_t n_seqs   = (uint32_t) src2->ne[3];
    స్థిరరాశి uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి దశాంశం    scale    = 1.0f / sqrtf((దశాంశం) s_v);
    uint32_t       scale_u32;
    జ్ఞాపకం_నకలు(&scale_u32, &scale, పరిమాణం(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_టోకెన్లు,
        n_seqs,
        s_v * h * n_టోకెన్లు * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_కొలత(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_కొలత(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_కొలత(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_కొలత(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_కొలత(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_కొలత(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_కొలత(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_కొలత(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0), ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, src2), ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 4, src4), ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 6, dst),
    };

    ఒకవేళ (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 7, dst_fuse));
    }

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

స్థిర std::optional<webgpu_encoded_op> ggml_webgpu_సమితి_అడ్డువరుసలు(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    ఒకవేళ (ggml_is_ఖాళీలు(src) || ggml_is_ఖాళీలు(idx)) {
        తిరిగి std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_సమితి_అడ్డువరుసలు_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_సమితి_అడ్డువరుసలు_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_కొలత(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)), (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)), (uint32_t) (idx->nb[0] / ggml_type_కొలత(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_కొలత(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_కొలత(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst));

    ఒకవేళ (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(3, ctx->సమితి_అడ్డువరుసలు_dev_error_buf, 0,
                                                            ctx->సమితి_అడ్డువరుసలు_dev_error_buf.Get_కొలత()));
    }

    uint32_t దారాలు;
    ఒకవేళ (ggml_is_quantized(dst->type)) {
        స్థిరరాశి uint32_t ఖండాలు_చొప్పున_అడ్డువరుస = src->ne[0] / ggml_blck_కొలత(dst->type);
        దారాలు =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_ఖండాలు ? (ఖండాలు_చొప్పున_అడ్డువరుస / 2) : ఖండాలు_చొప్పున_అడ్డువరుస);
    } లేకపోతే ఒకవేళ (decisions->vec4) {
        దారాలు = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } లేకపోతే {
        దారాలు = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(దారాలు, decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
స్థిర std::vector<wgpu::Constant_నమోదు> ggml_webgpu_wg_కొలత_నమోదు(uint32_t wg_కొలత) {
    std::vector<wgpu::Constant_నమోదు> constants(1);
    constants[0].కీలకం   = "wg_size";
    constants[0].value = wg_కొలత;
    తిరిగి constants;
}

స్థిర webgpu_encoded_op ggml_webgpu_get_అడ్డువరుసలు(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    స్థిరరాశి తార్కిక float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = శూన్యసూచిక;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    స్థిరరాశి uint32_t స్థానభ్రంశం_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type));
    స్థిరరాశి uint32_t స్థానభ్రంశం_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type));
    స్థిరరాశి uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type));
    స్థిరరాశి uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type));
    స్థిరరాశి uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type));

    స్థిరరాశి తార్కిక vec4_aligned = స్థానభ్రంశం_src % 4 == 0 && స్థానభ్రంశం_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_అడ్డువరుసలు_pipeline(shader_lib_ctx, vec4_aligned);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_get_అడ్డువరుసలు_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { స్థానభ్రంశం_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_కొలత(idx->type)),
                                     స్థానభ్రంశం_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_కొలత(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_కొలత(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_కొలత(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst) };

    uint32_t ఖండాలు_చొప్పున_అడ్డువరుస = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t మొత్తం_అడ్డువరుసలు     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t మొత్తం_దారాలు  = float_parallel ? ఖండాలు_చొప్పున_అడ్డువరుస * మొత్తం_అడ్డువరుసలు : మొత్తం_అడ్డువరుసలు;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(మొత్తం_దారాలు, decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర శూన్య ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_కొలత        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    స్థిరరాశి size_t dst_స్థానభ్రంశం           = ggml_webgpu_tensor_స్థానభ్రంశం(dst);
    స్థిరరాశి size_t q8_src1_align_స్థానభ్రంశం = ROUNDUP_POW2(
        dst_స్థానభ్రంశం + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
    స్థిరరాశి size_t q8_src1_binding_కొలత = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> q8_entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src1),
        ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_స్థానభ్రంశం, q8_src1_binding_కొలత)
    };

    స్వయంచాలక q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_కొలత     = q8_decisions->wg_కొలత;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    స్థిరరాశి uint32_t wg_చొప్పున_vec     = (src0->ne[0] / 4 + (q8_wg_కొలత - 1)) / q8_wg_కొలత;
    స్థిరరాశి uint32_t q8_మొత్తం_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_చొప్పున_vec;
    స్థిరరాశి uint32_t max_wg_చొప్పున_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension;
    compute_2d_workgroups(q8_మొత్తం_wg, max_wg_చొప్పున_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

స్థిర webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    తార్కిక use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    తార్కిక use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_లబ్ధం,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_కొలత              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_మాతృక = ctx->global_ctx->capabilities.supports_subgroup_మాతృక;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_కొలత        = ctx->global_ctx->capabilities.min_subgroup_కొలత;
    shader_lib_ctx.max_subgroup_కొలత        = ctx->global_ctx->capabilities.max_subgroup_కొలత;
    shader_lib_ctx.supports_dot_లబ్ధం     = ctx->global_ctx->capabilities.supports_dot_లబ్ధం;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    స్థిరరాశి తార్కిక src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    ఒకవేళ (use_mat_vec) {
        ఒకవేళ (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } లేకపోతే {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t స్థానభ్రంశం_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type));
    uint32_t స్థానభ్రంశం_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type));
    size_t   merged_స్థానభ్రంశం = 0;
    size_t   merged_కొలత   = 0;
    ఒకవేళ (src_overlap) {
        స్థిరరాశి ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_స్థానభ్రంశం = merged_range.స్థానభ్రంశం;
        merged_కొలత   = merged_range.size;
        స్థానభ్రంశం_src0   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src0, merged_range);
        స్థానభ్రంశం_src1   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { స్థానభ్రంశం_src0,
                                     స్థానభ్రంశం_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {};
    ఒకవేళ (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0));
        స్వయంచాలక & mmvq_qq8_నమోదు = dispatches[0].bind_సమూహం_entries[1];
        entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_నమోదు.స్థానభ్రంశం,
                                                            mmvq_qq8_నమోదు.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst));
    } లేకపోతే ఒకవేళ (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src0), merged_స్థానభ్రంశం, merged_కొలత));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst));
    } లేకపోతే {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    స్థిరరాశి uint32_t max_wg_చొప్పున_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension;

    ఒకవేళ (use_mat_vec) {
        స్వయంచాలక * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_చొప్పున_wg);
        uint32_t మొత్తం_wg      = output_groups * batches;
        compute_2d_workgroups(మొత్తం_wg, max_wg_చొప్పున_dim, wg_x, wg_y);
    } లేకపోతే {
        స్వయంచాలక * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        ఒకవేళ (decisions->use_subgroup_మాతృక) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_మాతృక_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_మాతృక_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } లేకపోతే {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_కొలత_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_కొలత_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t మొత్తం_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(మొత్తం_wg, max_wg_చొప్పున_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    తిరిగి ggml_backend_webgpu_build_multi(ctx, dispatches);
}

స్థిర webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    స్థిరరాశి uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    స్థిరరాశి uint32_t param_n_expert_వాడిన = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_కొలత(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_వాడిన,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src0),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, src0)),
        ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src1),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, src1)),
        ggml_webgpu_make_bind_సమూహం_నమోదు(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src2),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, src2)),
        ggml_webgpu_make_bind_సమూహం_నమోదు(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, dst),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    స్థిరరాశి uint32_t max_wg_చొప్పున_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_చొప్పున_wg);
    uint32_t       మొత్తం_wg       = output_groups * param_n_expert_వాడిన;
    compute_2d_workgroups(మొత్తం_wg, max_wg_చొప్పున_dim, wg_x, wg_y);

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
స్థిర uint32_t ggml_webgpu_mul_mat_id_max_tiles(స్థిరరాశి ggml_tensor * dst) {
    స్థిరరాశి uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    స్థిరరాశి uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    తిరిగి CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_కొలత_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
స్థిర size_t ggml_webgpu_mul_mat_id_tiles_nbytes(స్థిరరాశి ggml_tensor * dst) {
    తిరిగి (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * పరిమాణం(uint32_t);
}

స్థిర webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    ఒకవేళ (dst->ne[2] == 1) {
        తిరిగి ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    స్థిరరాశి uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    స్థిరరాశి uint32_t param_n_expert_వాడిన = (uint32_t) dst->ne[1];
    స్థిరరాశి uint32_t param_n_టోకెన్లు      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_కొలత(src2->type)),
        param_n_expert,
        param_n_expert_వాడిన,
        param_n_టోకెన్లు,
        (uint32_t) (src2->nb[1] / ggml_type_కొలత(src2->type)),
    };

    స్థిరరాశి size_t dst_స్థానభ్రంశం       = ggml_webgpu_tensor_స్థానభ్రంశం(dst);
    స్థిరరాశి size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * పరిమాణం(uint32_t);
    స్థిరరాశి size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    స్థిరరాశి size_t ids_cache_align_స్థానభ్రంశం = ROUNDUP_POW2(
        dst_స్థానభ్రంశం + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
    స్థిరరాశి size_t ids_tiles_align_స్థానభ్రంశం =
        ROUNDUP_POW2(ids_cache_align_స్థానభ్రంశం + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);

    స్థిరరాశి size_t ids_cache_bind_కొలత = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    స్థిరరాశి size_t ids_tiles_bind_కొలత = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::Bind_సమూహం_నమోదు> gather_entries = {
        ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src2),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, src2)),
        ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_స్థానభ్రంశం, ids_cache_bind_కొలత),
        ggml_webgpu_make_bind_సమూహం_నమోదు(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_స్థానభ్రంశం, ids_tiles_bind_కొలత),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_వాడిన,
        param_n_టోకెన్లు,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::Bind_సమూహం_నమోదు> main_entries = {
        ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src0),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, src0)),
        ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src1),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, src1)),
        ggml_webgpu_make_bind_సమూహం_నమోదు(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, dst),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, dst)),
        ggml_webgpu_make_bind_సమూహం_నమోదు(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_స్థానభ్రంశం, ids_cache_bind_కొలత),
        ggml_webgpu_make_bind_సమూహం_నమోదు(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_స్థానభ్రంశం, ids_tiles_bind_కొలత),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    స్వయంచాలక * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_కొలత_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_కొలత_N);

    స్థిరరాశి uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_కొలత_m;
    స్థిరరాశి uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    స్థిరరాశి uint32_t మొత్తం_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    తిరిగి ggml_backend_webgpu_build_multi(ctx, dispatches);
}

నిర్మాణం ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_సమూహం_నమోదు> entries;
    size_t                            kv_bind_స్థానభ్రంశం = 0;
    size_t                            kv_bind_కొలత   = 0;
    తార్కిక                              has_ముసుగు       = అబద్ధం;
    తార్కిక                              has_sinks      = అబద్ధం;
    తార్కిక                              kv_overlap     = అబద్ధం;
};

స్థిర తార్కిక ggml_webgpu_flash_attn_use_vec_మార్గం(స్థిరరాశి webgpu_global_context & global_ctx,
                                                స్థిరరాశి ggml_tensor *           Q,
                                                స్థిరరాశి ggml_tensor *           K,
                                                స్థిరరాశి ggml_tensor *           V) {
    స్థిరరాశి size_t storage_స్థానభ్రంశం_alignment = global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;

    స్థిరరాశి తార్కిక k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_స్థానభ్రంశం_alignment);
    స్థిరరాశి తార్కిక v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_స్థానభ్రంశం_alignment);

    స్థిరరాశి uint32_t k_vec_తల_align =
        ggml_is_quantized(K->type) ? ggml_blck_కొలత(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు;
    స్థిరరాశి uint32_t v_vec_తల_align =
        ggml_is_quantized(V->type) ? ggml_blck_కొలత(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు;
    స్థిరరాశి తార్కిక kv_vec_తల_dims_aligned = Q->ne[0] % k_vec_తల_align == 0 && V->ne[0] % v_vec_తల_align == 0;

    తిరిగి global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_తల_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

స్థిర ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    ముసుగు,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    దశాంశం scale         = ggml_get_op_params_f32(dst, 0);
    దశాంశం max_bias      = ggml_get_op_params_f32(dst, 1);
    దశాంశం logit_softcap = ggml_get_op_params_f32(dst, 2);
    ఒకవేళ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    దశాంశం n_తల_log2 = దశాంశం(1u << (uint32_t) floor(log2(Q->ne[2])));
    దశాంశం m0          = powf(2.0f, -(max_bias) / n_తల_log2);
    దశాంశం m1          = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = ముసుగు;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_మాతృక = ctx->global_ctx->capabilities.supports_subgroup_మాతృక;
    op.shader_lib_ctx.max_wg_కొలత              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
    op.shader_lib_ctx.wg_mem_హద్దు_బైట్లు       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_కొలత;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_కొలత        = ctx->global_ctx->capabilities.min_subgroup_కొలత;
    op.shader_lib_ctx.max_subgroup_కొలత        = ctx->global_ctx->capabilities.max_subgroup_కొలత;

    op.has_ముసుగు   = ముసుగు != శూన్యసూచిక;
    op.has_sinks  = sinks != శూన్యసూచిక;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t స్థానభ్రంశం_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_కొలత(K->type));
    uint32_t స్థానభ్రంశం_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_కొలత(V->type));
    ఒకవేళ (op.kv_overlap) {
        స్థిరరాశి ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_స్థానభ్రంశం                                   = merged_range.స్థానభ్రంశం;
        op.kv_bind_కొలత                                     = merged_range.size;
        స్థానభ్రంశం_k                                            = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(K, merged_range);
        స్థానభ్రంశం_v                                            = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_కొలత(Q->type)),
        స్థానభ్రంశం_k,
        స్థానభ్రంశం_v,
        op.has_ముసుగు ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ముసుగు) / ggml_type_కొలత(ముసుగు->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_కొలత(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_కొలత(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_కొలత(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_కొలత(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_కొలత(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_కొలత(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_కొలత(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_కొలత(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_కొలత(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_కొలత(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_ముసుగు ? (uint32_t) (ముసుగు->nb[3] / ggml_type_కొలత(ముసుగు->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_తల_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, Q),
    };
    ఒకవేళ (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(K), op.kv_bind_స్థానభ్రంశం, op.kv_bind_కొలత));
    } లేకపోతే {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, V));
    }
    uint32_t binding_సూచీ = op.kv_overlap ? 2u : 3u;
    ఒకవేళ (op.has_ముసుగు) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_సూచీ++, ముసుగు));
    }
    ఒకవేళ (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_సూచీ++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_సూచీ++, dst));

    తిరిగి op;
}

స్థిర uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    స్థిరరాశి uint64_t kv_పరిధి = (uint64_t) kv_tile;
    అయితే ((2u * nwg * kv_పరిధి) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    తిరిగి std::min(nwg, vec_nwg_cap);
}

స్థిర webgpu_encoded_op ggml_webgpu_flash_attn_ప్రత్యక్ష(webgpu_context & ctx, స్థిరరాశి ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    స్వయంచాలక *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_చొప్పున_తల = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_చొప్పున_తల * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

స్థిర webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             ముసుగు,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::బఫరు_2 blk_buf         = {};
    uint64_t     blk_కొలత_బైట్లు  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_లెక్క = 0;

    స్థిరరాశి uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_కొలత;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    స్థిరరాశి uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    స్థిరరాశి తార్కిక     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_పునాది  = 0;
    uint64_t     tmp_కొలత_బైట్లు  = 0;
    wgpu::బఫరు_2 tmp_buf         = {};
    uint64_t     tmp_bind_స్థానభ్రంశం = 0;
    uint64_t     tmp_bind_కొలత   = 0;
    స్థిరరాశి size_t align_బైట్లు     = ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;
    స్థిరరాశి size_t dst_స్థానభ్రంశం      = ggml_webgpu_tensor_స్థానభ్రంశం(dst);
    size_t       scratch_స్థానభ్రంశం  = ROUNDUP_POW2(dst_స్థానభ్రంశం + ggml_nbytes(dst), align_బైట్లు);

    ఒకవేళ (use_vec_reduce) {
        స్థిరరాశి uint64_t tmp_దత్తాంశం_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        స్థిరరాశి uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_పునాది                 = tmp_దత్తాంశం_elems;
        tmp_కొలత_బైట్లు =
            ROUNDUP_POW2((tmp_దత్తాంశం_elems + tmp_stats_elems) * పరిమాణం(దశాంశం), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_పునాది <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_స్థానభ్రంశం = scratch_స్థానభ్రంశం;
        tmp_bind_కొలత   = tmp_కొలత_బైట్లు;
        scratch_స్థానభ్రంశం  = ROUNDUP_POW2(scratch_స్థానభ్రంశం + tmp_కొలత_బైట్లు, align_బైట్లు);
    } లేకపోతే {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_కొలత_బైట్లు  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_స్థానభ్రంశం = scratch_స్థానభ్రంశం;
        tmp_bind_కొలత   = tmp_కొలత_బైట్లు;
        scratch_స్థానభ్రంశం  = ROUNDUP_POW2(scratch_స్థానభ్రంశం + tmp_కొలత_బైట్లు, align_బైట్లు);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::Bind_సమూహం_నమోదు> blk_entries;
    ఒకవేళ (op.has_ముసుగు) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        స్థిరరాశి uint32_t stride_mask3 = (uint32_t) (ముసుగు->nb[3] / ggml_type_కొలత(ముసుగు->type));
        blk_batch_లెక్క             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        స్థిరరాశి uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_లెక్క;
        blk_కొలత_బైట్లు              = ROUNDUP_POW2(blk_elems * పరిమాణం(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        స్థిరరాశి ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ముసుగు) / ggml_type_కొలత(ముసుగు->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(ముసుగు),
                                              ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, ముసుగు),
                                              ggml_webgpu_tensor_binding_కొలత(ctx, ముసుగు)),
            ggml_webgpu_make_bind_సమూహం_నమోదు(1, blk_buf, scratch_స్థానభ్రంశం, blk_కొలత_బైట్లు),
        };
        scratch_స్థానభ్రంశం = ROUNDUP_POW2(scratch_స్థానభ్రంశం + blk_కొలత_బైట్లు, align_బైట్లు);
    }

    std::vector<uint32_t> విభజించు_params = op.params;
    ఒకవేళ (op.has_ముసుగు) {
        విభజించు_params.push_back(0u);                     // blk_base
        విభజించు_params.push_back(blk_nblk0);              // blk_nblk0
        విభజించు_params.push_back(blk_nblk1);              // blk_nblk1
    }
    విభజించు_params.push_back(0u);                         // tmp_data_base
    విభజించు_params.push_back((uint32_t) tmp_stats_పునాది);  // tmp_stats_base
    విభజించు_params.push_back(nwg);                        // nwg

    std::vector<wgpu::Bind_సమూహం_నమోదు> విభజించు_entries = {
        ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, Q),
                                          ggml_webgpu_tensor_binding_కొలత(ctx, Q)),
    };
    ఒకవేళ (op.kv_overlap) {
        విభజించు_entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(K), op.kv_bind_స్థానభ్రంశం, op.kv_bind_కొలత));
    } లేకపోతే {
        విభజించు_entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, K),
                                                                  ggml_webgpu_tensor_binding_కొలత(ctx, K)));
        విభజించు_entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, V),
                                                                  ggml_webgpu_tensor_binding_కొలత(ctx, V)));
    }
    uint32_t విభజించు_binding_సూచీ = op.kv_overlap ? 2u : 3u;
    ఒకవేళ (op.has_ముసుగు) {
        విభజించు_entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(విభజించు_binding_సూచీ++, ggml_webgpu_tensor_buf(ముసుగు),
                                                                  ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, ముసుగు),
                                                                  ggml_webgpu_tensor_binding_కొలత(ctx, ముసుగు)));
    }
    ఒకవేళ (op.has_sinks) {
        విభజించు_entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(విభజించు_binding_సూచీ++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_కొలత(ctx, sinks)));
    }
    ఒకవేళ (op.has_ముసుగు) {
        విభజించు_entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(విభజించు_binding_సూచీ++, blk_buf, blk_entries[1].స్థానభ్రంశం, blk_కొలత_బైట్లు));
    }
    విభజించు_entries.push_back(
        ggml_webgpu_make_bind_సమూహం_నమోదు(విభజించు_binding_సూచీ++, tmp_buf, tmp_bind_స్థానభ్రంశం, tmp_bind_కొలత));
    విభజించు_entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(విభజించు_binding_సూచీ++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, dst),
                                                              ggml_webgpu_tensor_binding_కొలత(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::Bind_సమూహం_నమోదు> reduce_entries;
    ఒకవేళ (use_vec_reduce) {
        స్థిరరాశి uint32_t reduce_sg_కొలత = ctx->global_ctx->capabilities.max_subgroup_కొలత;
        స్థిరరాశి uint32_t reduce_wg_కొలత = std::max(
            reduce_sg_కొలత,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_కొలత,
                                          ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_కొలత                    = reduce_wg_కొలత;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_పునాది,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_సమూహం_నమోదు(0, tmp_buf, tmp_bind_స్థానభ్రంశం, tmp_కొలత_బైట్లు),
            ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, dst),
                                              ggml_webgpu_tensor_binding_కొలత(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    స్థిరరాశి uint64_t విభజించు_wg_మొత్తం = (uint64_t) wg_x * nwg;
    GGML_ASSERT(విభజించు_wg_మొత్తం <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    ఒకవేళ (op.has_ముసుగు) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_లెక్క }
        });
    }
    dispatches.push_back({
        pipeline, std::move(విభజించు_params), std::move(విభజించు_entries), { (uint32_t) విభజించు_wg_మొత్తం, 1u }
    });
    ఒకవేళ (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    తిరిగి ggml_backend_webgpu_build_multi(ctx, dispatches);
}

స్థిర webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    ముసుగు,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, ముసుగు, sinks, dst);
    ఒకవేళ (ggml_webgpu_flash_attn_use_vec_మార్గం(ctx->global_ctx, Q, K, V)) {
        తిరిగి ggml_webgpu_flash_attn_vec(ctx, Q, K, V, ముసుగు, sinks, dst, std::move(op));
    }
    తిరిగి ggml_webgpu_flash_attn_ప్రత్యక్ష(ctx, op);
}

స్థిర webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    తార్కిక is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = శూన్యసూచిక;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    స్వయంచాలక *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    స్థిరరాశి తార్కిక inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_కొలత(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    ఒకవేళ (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        ఎంపిక (unary_op) {
            స్థితి GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    దశాంశం alpha_n = ggml_get_op_params_f32(dst, 1);
                    దశాంశం alpha_p = ggml_get_op_params_f32(dst, 2);
                    దశాంశం beta    = ggml_get_op_params_f32(dst, 3);
                    దశాంశం eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    ఆపు;
                }
            అప్రమేయ:
                ఆపు;
        }
    } లేకపోతే ఒకవేళ (dst->op == GGML_OP_CLAMP) {
        దశాంశం clamp_min = ggml_get_op_params_f32(dst, 0);
        దశాంశం clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } లేకపోతే ఒకవేళ (dst->op == GGML_OP_FILL) {
        దశాంశం fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, effective_src),
    };
    ఒకవేళ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_ద్విమాన_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    స్థిరరాశి తార్కిక      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_ద్విమాన_pipeline(shader_lib_ctx, src_overlap);
    స్వయంచాలక *          decisions   = static_cast<ggml_webgpu_ద్విమాన_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_స్థానభ్రంశం = ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src0);
    size_t src1_webgpu_tensor_align_స్థానభ్రంశం = ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src1);

    uint32_t స్థానభ్రంశం_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type));
    uint32_t స్థానభ్రంశం_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type));
    size_t   merged_స్థానభ్రంశం = 0;
    size_t   merged_కొలత   = 0;
    ఒకవేళ (decisions->src_overlap) {
        స్థిరరాశి ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_స్థానభ్రంశం = merged_range.స్థానభ్రంశం;
        merged_కొలత   = merged_range.size;
        స్థానభ్రంశం_src0   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src0, merged_range);
        స్థానభ్రంశం_src1   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        స్థానభ్రంశం_src0,
        స్థానభ్రంశం_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries;

    ఒకవేళ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src0), merged_స్థానభ్రంశం, merged_కొలత));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst));
    } లేకపోతే {
        entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_స్థానభ్రంశం,
                                                            ggml_webgpu_tensor_binding_కొలత(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_స్థానభ్రంశం,
                                                            ggml_webgpu_tensor_binding_కొలత(ctx, src1)));
        ఒకవేళ (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_కూడు_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_కూడు_id_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_కొలత(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_కొలత(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_కొలత(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, src2));

    ఒకవేళ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       మొత్తం_wg       = ggml_nrows(dst);
    స్థిరరాశి uint32_t max_wg_చొప్పున_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension;
    compute_2d_workgroups(మొత్తం_wg, max_wg_చొప్పున_dim, wg_x, wg_y);

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    ఒకవేళ (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        తిరిగి {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    స్థిరరాశి తార్కిక      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    స్వయంచాలక *          decisions   = static_cast<ggml_webgpu_ద్విమాన_shader_decisions *>(pipeline.context.get());

    uint32_t స్థానభ్రంశం_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type));
    uint32_t స్థానభ్రంశం_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type));
    size_t   merged_స్థానభ్రంశం = 0;
    size_t   merged_కొలత   = 0;
    ఒకవేళ (decisions->src_overlap) {
        ఒకవేళ (ggml_nbytes(src0) == 0) {
            merged_స్థానభ్రంశం = ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src1);
            merged_కొలత   = ggml_webgpu_tensor_binding_కొలత(ctx, src1);
            స్థానభ్రంశం_src0   = 0;
            స్థానభ్రంశం_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type));
        } లేకపోతే ఒకవేళ (ggml_nbytes(src1) == 0) {
            merged_స్థానభ్రంశం = ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src0);
            merged_కొలత   = ggml_webgpu_tensor_binding_కొలత(ctx, src0);
            స్థానభ్రంశం_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type));
            స్థానభ్రంశం_src1   = 0;
        } లేకపోతే {
            స్థిరరాశి ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_స్థానభ్రంశం = merged_range.స్థానభ్రంశం;
            merged_కొలత   = merged_range.size;
            స్థానభ్రంశం_src0   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src0, merged_range);
            స్థానభ్రంశం_src1   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     స్థానభ్రంశం_src0,
                                     స్థానభ్రంశం_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_కొలత(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {};
    ఒకవేళ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src0), merged_స్థానభ్రంశం, merged_కొలత));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst));
    } లేకపోతే {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ne, decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_కొలత(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ne, decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    ఒకవేళ (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } లేకపోతే ఒకవేళ (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } లేకపోతే {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t స్థానభ్రంశం_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_కొలత(rn_src->type));
    uint32_t స్థానభ్రంశం_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_కొలత(mul_src->type));
    size_t merged_స్థానభ్రంశం = 0;
    size_t merged_కొలత   = 0;

    std::vector<uint32_t> params = {
        స్థానభ్రంశం_rn_src,
        స్థానభ్రంశం_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_కొలత(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_కొలత(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_కొలత(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_కొలత(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_కొలత(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_కొలత(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
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

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    స్థిరరాశి తార్కిక      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    స్వయంచాలక *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    ఒకవేళ (decisions->src_overlap) {
        స్థిరరాశి ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_స్థానభ్రంశం  = merged_range.స్థానభ్రంశం;
        merged_కొలత    = merged_range.size;
        స్థానభ్రంశం_rn_src  = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(rn_src, merged_range);
        స్థానభ్రంశం_mul_src = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(mul_src, merged_range);
        params[0]      = స్థానభ్రంశం_rn_src;
        params[1]      = స్థానభ్రంశం_mul_src;
    }

    ఒకవేళ (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, mul_src));
    } లేకపోతే ఒకవేళ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(rn_src), merged_స్థానభ్రంశం, merged_కొలత));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst));
    } లేకపోతే {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x,
                          wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_అడ్డువరుస_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_అడ్డువరుస_norm_pipeline(shader_lib_ctx);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src) };
    ఒకవేళ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x,
                          wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    స్థిరరాశి తార్కిక inplace         = decisions->inplace;
    స్థిరరాశి పూర్ణాంకం  has_freq_గుణకం = (src2 != శూన్యసూచిక);

    స్థిరరాశి పూర్ణాంకం n_dims     = ((int32_t *) dst->op_params)[1];
    స్థిరరాశి పూర్ణాంకం mode       = ((int32_t *) dst->op_params)[2];
    స్థిరరాశి పూర్ణాంకం n_ctx_orig = ((int32_t *) dst->op_params)[4];
    స్థిరరాశి పూర్ణాంకం n_offs     = ((int32_t *) dst->op_params)[15];

    దశాంశం freq_పునాది;
    దశాంశం freq_scale;
    దశాంశం ext_గుణకం;
    దశాంశం attn_గుణకం;
    దశాంశం beta_fast;
    దశాంశం beta_slow;
    జ్ఞాపకం_నకలు(&freq_పునాది, (int32_t *) dst->op_params + 5, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&freq_scale, (int32_t *) dst->op_params + 6, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&ext_గుణకం, (int32_t *) dst->op_params + 7, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&attn_గుణకం, (int32_t *) dst->op_params + 8, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_fast, (int32_t *) dst->op_params + 9, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_slow, (int32_t *) dst->op_params + 10, పరిమాణం(దశాంశం));

    పూర్ణాంకం sections[4];
    జ్ఞాపకం_నకలు(sections, (int32_t *) dst->op_params + 11, 4 * పరిమాణం(పూర్ణాంకం));

    దశాంశం theta_scale = powf(freq_పునాది, -2.0f / n_dims);

    దశాంశం corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_పునాది, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)),
        src2 != శూన్యసూచిక ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_కొలత(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_గుణకం),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_గుణకం),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries     = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    ఒకవేళ (has_freq_గుణకం) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 2, src2));
    }
    ఒకవేళ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    స్థిరరాశి తార్కిక      src_overlap = src1 != శూన్యసూచిక && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    స్వయంచాలక * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    స్థిరరాశి పూర్ణాంకం విభజించు = (src1 != శూన్యసూచిక);

    uint32_t స్థానభ్రంశం_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type));
    uint32_t స్థానభ్రంశం_src1 =
        src1 != శూన్యసూచిక ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)) : 0;
    size_t merged_స్థానభ్రంశం = 0;
    size_t merged_కొలత   = 0;
    ఒకవేళ (decisions->src_overlap) {
        స్థిరరాశి ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_స్థానభ్రంశం = merged_range.స్థానభ్రంశం;
        merged_కొలత   = merged_range.size;
        స్థానభ్రంశం_src0   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src0, merged_range);
        స్థానభ్రంశం_src1   = ggml_webgpu_tensor_merged_element_స్థానభ్రంశం(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        స్థానభ్రంశం_src0,
        స్థానభ్రంశం_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
        src1 != శూన్యసూచిక ? (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        src1 != శూన్యసూచిక ? (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        src1 != శూన్యసూచిక ? (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  //హద్దు
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries;
    uint32_t                          dst_binding = 1;
    ఒకవేళ (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_సమూహం_నమోదు(0, ggml_webgpu_tensor_buf(src0), merged_స్థానభ్రంశం, merged_కొలత));
    } లేకపోతే {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src0));
    }
    ఒకవేళ (విభజించు && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = శూన్యసూచిక;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src) };

    ఒకవేళ (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    స్థిరరాశి తార్కిక inplace     = decisions->inplace;
    స్థిరరాశి పూర్ణాంకం  has_ముసుగు    = (src1 != శూన్యసూచిక);
    స్థిరరాశి పూర్ణాంకం  has_sink    = (src2 != శూన్యసూచిక);
    దశాంశం      max_bias    = ggml_get_op_params_f32(dst, 1);
    దశాంశం      n_తల_log2 = దశాంశం(1u << (uint32_t) floor(log2(src0->ne[2])));
    దశాంశం      m0          = powf(2.0f, -(max_bias) / n_తల_log2);
    దశాంశం      m1          = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_కొలత(src0->type)),
        has_ముసుగు ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_కొలత(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_కొలత(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_కొలత(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_కొలత(src0->type)),
        has_ముసుగు ? (uint32_t) (src1->nb[1] / ggml_type_కొలత(src1->type)) : 0,
        has_ముసుగు ? (uint32_t) (src1->nb[2] / ggml_type_కొలత(src1->type)) : 0,
        has_ముసుగు ? (uint32_t) (src1->nb[3] / ggml_type_కొలత(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_ముసుగు ? (uint32_t) src1->ne[2] : 0,
        has_ముసుగు ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_తల_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries     = { ggml_webgpu_make_bind_సమూహం_నమోదు(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src0),
        ggml_webgpu_tensor_binding_కొలత(ctx, src0)) };
    uint32_t                          binding_అంకె = 1;
    ఒకవేళ (has_ముసుగు) {
        entries.push_back(ggml_webgpu_make_bind_సమూహం_నమోదు(binding_అంకె, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, src1),
                                                            ggml_webgpu_tensor_binding_కొలత(ctx, src1)));
        binding_అంకె++;
    }
    ఒకవేళ (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె, src2));
        binding_అంకె++;
    }
    ఒకవేళ (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, binding_అంకె, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = ggml_nrows(dst);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

స్థిర webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    తార్కిక is_శిఖరం_k = dst->op == GGML_OP_శిఖరం_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = శూన్యసూచిక;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
    shader_lib_ctx.wg_mem_హద్దు_బైట్లు = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_కొలత;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    స్వయంచాలక * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_విలీనం_pipeline = ctx->shader_lib->get_argsort_విలీనం_pipeline(shader_lib_ctx);

    స్థిరరాశి uint32_t src_ne0 = (uint32_t) src->ne[0];
    స్థిరరాశి uint32_t nrows   = (uint32_t) ggml_nrows(src);
    స్థిరరాశి uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_కొలత);
    స్థిరరాశి uint32_t ఖండం_కొలత_3 =
        is_శిఖరం_k ? std::min(argsort_decisions->wg_కొలత, (uint32_t) dst->ne[0]) : argsort_decisions->wg_కొలత;
    uint32_t బహిర్గతం_ne0 = src_ne0;
    ఒకవేళ (is_శిఖరం_k) {
        ఒకవేళ (npr > 1) {
            స్థిరరాశి uint32_t చివరి_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_కొలత;
            బహిర్గతం_ne0                  = (npr - 1) * ఖండం_కొలత_3 + std::min(చివరి_tile, ఖండం_కొలత_3);
        } లేకపోతే {
            బహిర్గతం_ne0 = ఖండం_కొలత_3;
        }
    }

    uint32_t విలీనం_len    = ఖండం_కొలత_3;
    uint32_t విలీనం_passes = 0;
    అయితే (విలీనం_len < బహిర్గతం_ne0) {
        విలీనం_len <<= 1;
        విలీనం_passes++;
    }

    స్థిరరాశి తార్కిక ప్రారంభం_లోపల_tmp = (విలీనం_passes % 2) == 1;

    స్థిరరాశి size_t dst_స్థానభ్రంశం = ggml_webgpu_tensor_స్థానభ్రంశం(dst);
    స్థిరరాశి size_t idx_nbytes = బహిర్గతం_ne0 * ggml_nrows(dst) * పరిమాణం(int32_t);
    స్థిరరాశి size_t tmp_స్థానభ్రంశం =
        ROUNDUP_POW2(dst_స్థానభ్రంశం + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment);
    స్థిరరాశి size_t tmp_binding_కొలత = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    స్థిరరాశి size_t dst_binding_కొలత =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    స్థిరరాశి uint32_t స్థానభ్రంశం_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type));
    స్థిరరాశి uint32_t స్థానభ్రంశం_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type));
    స్థిరరాశి uint32_t స్థానభ్రంశం_tmp  = 0;
    స్థిరరాశి uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type));
    స్థిరరాశి uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type));
    స్థిరరాశి uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type));
    స్థిరరాశి uint32_t stride_idx1 = బహిర్గతం_ne0;
    స్థిరరాశి uint32_t stride_idx2 = బహిర్గతం_ne0 * (uint32_t) dst->ne[1];
    స్థిరరాశి uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    స్థిరరాశి uint32_t init_స్థానభ్రంశం       = ప్రారంభం_లోపల_tmp ? స్థానభ్రంశం_tmp : స్థానభ్రంశం_dst;
    స్థిరరాశి size_t   init_align_స్థానభ్రంశం = ప్రారంభం_లోపల_tmp ? tmp_స్థానభ్రంశం : ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, dst);
    స్థిరరాశి size_t   init_binding_కొలత = ప్రారంభం_లోపల_tmp ? tmp_binding_కొలత : dst_binding_కొలత;

    std::vector<uint32_t> init_params = {
        స్థానభ్రంశం_src,  init_స్థానభ్రంశం, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], బహిర్గతం_ne0,
        ఖండం_కొలత_3,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    స్థిరరాశి uint32_t మొత్తం_wg_init  = npr * nrows;
    స్థిరరాశి uint32_t max_wg_చొప్పున_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension;
    compute_2d_workgroups(మొత్తం_wg_init, max_wg_చొప్పున_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::Bind_సమూహం_నమోదు> init_entries = {
        ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
        ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(dst), init_align_స్థానభ్రంశం, init_binding_కొలత)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    ఒకవేళ (విలీనం_passes == 0) {
        తిరిగి ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    తార్కిక     లోపల_is_tmp = ప్రారంభం_లోపల_tmp;
    uint32_t len       = ఖండం_కొలత_3;
    అయితే (len < బహిర్గతం_ne0) {
        స్థిరరాశి uint32_t nm = CEIL_DIV(బహిర్గతం_ne0, 2 * len);

        స్థిరరాశి తార్కిక     బహిర్గతం_is_tmp  = !లోపల_is_tmp;
        స్థిరరాశి uint32_t స్థానభ్రంశం_లోపల   = లోపల_is_tmp ? స్థానభ్రంశం_tmp : స్థానభ్రంశం_dst;
        స్థిరరాశి uint32_t స్థానభ్రంశం_బహిర్గతం  = బహిర్గతం_is_tmp ? స్థానభ్రంశం_tmp : స్థానభ్రంశం_dst;
        స్థిరరాశి size_t   align_లోపల    = లోపల_is_tmp ? tmp_స్థానభ్రంశం : ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, dst);
        స్థిరరాశి size_t   align_బహిర్గతం   = బహిర్గతం_is_tmp ? tmp_స్థానభ్రంశం : ggml_webgpu_tensor_align_స్థానభ్రంశం(ctx, dst);
        స్థిరరాశి size_t   కొలత_లోపల     = లోపల_is_tmp ? tmp_binding_కొలత : dst_binding_కొలత;
        స్థిరరాశి size_t   కొలత_బహిర్గతం    = బహిర్గతం_is_tmp ? tmp_binding_కొలత : dst_binding_కొలత;
        స్థిరరాశి uint32_t శిఖరం_k_బహిర్గతం   = (is_శిఖరం_k && nm == 1) ? (uint32_t) dst->ne[0] : బహిర్గతం_ne0;
        స్థిరరాశి uint32_t stride_out1 = శిఖరం_k_బహిర్గతం;
        స్థిరరాశి uint32_t stride_out2 = శిఖరం_k_బహిర్గతం * (uint32_t) dst->ne[1];
        స్థిరరాశి uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> విలీనం_params = { స్థానభ్రంశం_src,
                                               స్థానభ్రంశం_లోపల,
                                               స్థానభ్రంశం_బహిర్గతం,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               బహిర్గతం_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               శిఖరం_k_బహిర్గతం,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::Bind_సమూహం_నమోదు> విలీనం_entries = {
            ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
            ggml_webgpu_make_bind_సమూహం_నమోదు(1, ggml_webgpu_tensor_buf(dst), align_లోపల, కొలత_లోపల),
            ggml_webgpu_make_bind_సమూహం_నమోదు(2, ggml_webgpu_tensor_buf(dst), align_బహిర్గతం, కొలత_బహిర్గతం)
        };

        uint32_t       wg_x_విలీనం;
        uint32_t       wg_y_విలీనం;
        స్థిరరాశి uint32_t మొత్తం_wg_విలీనం = nm * nrows;
        compute_2d_workgroups(మొత్తం_wg_విలీనం, max_wg_చొప్పున_dim, wg_x_విలీనం, wg_y_విలీనం);

        dispatches.push_back({
            argsort_విలీనం_pipeline, std::move(విలీనం_params), std::move(విలీనం_entries), { wg_x_విలీనం, wg_y_విలీనం }
        });

        len <<= 1;
        లోపల_is_tmp = !లోపల_is_tmp;
    }

    తిరిగి ggml_backend_webgpu_build_multi(ctx, dispatches);
}

స్థిర webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = శూన్యసూచిక;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        మొత్తం_wg = ggml_nrows(dst);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర webgpu_encoded_op ggml_webgpu_కూడిక_అడ్డువరుసలు(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    తార్కిక                  మొత్తం_కూడిక = dst->op == GGML_OP_కూడిక;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),
                                     మొత్తం_కూడిక ? 0 : (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)),
                                     మొత్తం_కూడిక ? 0 : (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)),
                                     మొత్తం_కూడిక ? 0 : (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)),
                                     మొత్తం_కూడిక ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     మొత్తం_కూడిక ? 1 : (uint32_t) src->ne[1],
                                     మొత్తం_కూడిక ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_కూడిక_అడ్డువరుసలు_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = మొత్తం_కూడిక ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);
    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

స్థిర తార్కిక ggml_webgpu_can_fuse_rms_norm_mul(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        తిరిగి అబద్ధం;
    }

    // additional constraints specific to this fusion
    స్థిరరాశి ggml_tensor * rms_norm = cgraph->nodes[కణుపు_idx];
    స్థిరరాశి ggml_tensor * mul      = cgraph->nodes[కణుపు_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    ఒకవేళ (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    ఒకవేళ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        తిరిగి అబద్ధం;
    }
    // rms_norm shader assumes contiguous rows
    ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(mul->src[0]) || !ggml_is_contiguous_అడ్డువరుసలు(mul->src[1])) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_webgpu_can_fuse_gdn_cache(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, పూర్ణాంకం & అంకె_encoded_ops) {
    స్థిరరాశి ggml_tensor * gdn = cgraph->nodes[కణుపు_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ఒకవేళ (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_tensor * src_v     = gdn->src[2];
    స్థిరరాశి int64_t       S_v       = src_v->ne[0];
    స్థిరరాశి int64_t       H         = src_v->ne[1];
    స్థిరరాశి int64_t       n_టోకెన్లు  = src_v->ne[2];
    స్థిరరాశి int64_t       n_seqs    = src_v->ne[3];
    స్థిరరాశి int64_t       D         = S_v * S_v * H;
    స్థిరరాశి int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    స్థిరరాశి int64_t       n_written = std::min<int64_t>(n_టోకెన్లు, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    స్థిరరాశి size_t తోక_off = ggml_అడ్డువరుస_కొలత(GGML_TYPE_F32, S_v * H * n_టోకెన్లు * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    స్థిరరాశి ggml_tensor * cpy     = శూన్యసూచిక;
    పూర్ణాంకం                 cpy_idx = 0;
    కొరకు (పూర్ణాంకం j = కణుపు_idx + 1; j < cgraph->n_nodes && cpy == శూన్యసూచిక; ++j) {
        స్థిరరాశి ggml_tensor * n = cgraph->nodes[j];
        ఒకవేళ (ggml_op_is_ఖాళీలు(n->op) || ggml_is_ఖాళీలు(n)) {
            కొనసాగు;
        }
        ఒకవేళ (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            తిరిగి అబద్ధం;
        }
        cpy     = n;
        cpy_idx = j;
    }
    ఒకవేళ (cpy == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    స్థిరరాశి ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ఒకవేళ (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != తోక_off ||
        !ggml_is_contiguous(cpy_src)) {
        తిరిగి అబద్ధం;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    స్థిరరాశి std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    ఒకవేళ (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->దత్తాంశం == శూన్యసూచిక ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_కొలత(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_అడ్డువరుస_కొలత(GGML_TYPE_F32, D)) {
        తిరిగి అబద్ధం;
    }

    అంకె_encoded_ops = cpy_idx - కణుపు_idx + 1;

    తిరిగి నిజం;
}

స్థిర webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    స్థిరరాశి uint32_t        బహుళకం_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_కొలత(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_కొలత(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_కొలత(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_కొలత(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_కొలత(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_కొలత(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_కొలత(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_కొలత(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     బహుళకం_flags };

    std::vector<wgpu::Bind_సమూహం_నమోదు> entries = { ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_సమూహం_నమోదు(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_కొలత = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    స్వయంచాలక *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t మొత్తం_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_కొలత);
    compute_2d_workgroups(మొత్తం_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension, wg_x, wg_y);

    తిరిగి ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
స్థిర std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           పూర్ణాంకం            కణుపు_idx,
                                                           పూర్ణాంకం &          అంకె_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  కణుపు  = nodes[కణుపు_idx];

    ఒకవేళ (ggml_is_ఖాళీలు(కణుపు)) {
        తిరిగి std::nullopt;
    }
    ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        తిరిగి std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << కణుపు << ", " << ggml_op_name(కణుపు->op) << ")");

    ggml_tensor * src0 = కణుపు->src[0];
    ggml_tensor * src1 = కణుపు->src[1];
    ggml_tensor * src2 = కణుపు->src[2];

    ఎంపిక (కణుపు->op) {
            // no-ops
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_RESHAPE:
            తిరిగి std::nullopt;
        స్థితి GGML_OP_DUP:
        స్థితి GGML_OP_CPY:
        స్థితి GGML_OP_CONT:
            తిరిగి ggml_webgpu_cpy(ctx, src0, కణుపు);
        స్థితి GGML_OP_సమితి:
            తిరిగి ggml_webgpu_సమితి(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            తిరిగి ggml_webgpu_సమితి_అడ్డువరుసలు(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            తిరిగి ggml_webgpu_get_అడ్డువరుసలు(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_MUL_MAT:
            తిరిగి ggml_webgpu_mul_mat(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_MUL_MAT_ID:
            తిరిగి ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, కణుపు);
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            తిరిగి ggml_webgpu_flash_attn(ctx, src0, src1, src2, కణుపు->src[3], కణుపు->src[4], కణుపు);
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_DIV:
            తిరిగి ggml_webgpu_ద్విమాన_op(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_కూడు_ID:
            తిరిగి ggml_webgpu_కూడు_id(ctx, src0, src1, src2, కణుపు);
        స్థితి GGML_OP_CONCAT:
            తిరిగి ggml_webgpu_concat(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_REPEAT:
            తిరిగి ggml_webgpu_repeat(ctx, src0, కణుపు);
        స్థితి GGML_OP_RMS_NORM:
            ఒకవేళ (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, కణుపు_idx)) {
                అంకె_encoded_ops        = 2;
                ggml_tensor * mul_కణుపు = nodes[కణుపు_idx + 1];
                తిరిగి ggml_webgpu_rms_norm_mul(ctx, src0, కణుపు, mul_కణుపు->src[0], mul_కణుపు->src[1], mul_కణుపు);
            } లేకపోతే {
                తిరిగి ggml_webgpu_అడ్డువరుస_norm(ctx, src0, కణుపు);
            }
        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_L2_NORM:
            తిరిగి ggml_webgpu_అడ్డువరుస_norm(ctx, src0, కణుపు);
        స్థితి GGML_OP_ROPE:
            తిరిగి ggml_webgpu_rope(ctx, src0, src1, src2, కణుపు);
        స్థితి GGML_OP_GLU:
            తిరిగి ggml_webgpu_glu(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_SCALE:
            తిరిగి ggml_webgpu_scale(ctx, src0, కణుపు);
        స్థితి GGML_OP_SOFT_MAX:
            తిరిగి ggml_webgpu_soft_max(ctx, src0, src1, src2, కణుపు);
        స్థితి GGML_OP_UNARY:
        స్థితి GGML_OP_CLAMP:
        స్థితి GGML_OP_FILL:
        స్థితి GGML_OP_LOG:
        స్థితి GGML_OP_SQR:
        స్థితి GGML_OP_SQRT:
        స్థితి GGML_OP_SIN:
        స్థితి GGML_OP_COS:
        స్థితి GGML_OP_DIAG:
        స్థితి GGML_OP_TRI:
            తిరిగి ggml_webgpu_unary_op(ctx, src0, కణుపు);
        స్థితి GGML_OP_SOLVE_TRI:
            తిరిగి ggml_webgpu_solve_tri(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_SSM_CONV:
            తిరిగి ggml_webgpu_ssm_conv(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_SSM_పరిశీలన:
            తిరిగి ggml_webgpu_ssm_పరిశీలన(ctx, src0, src1, src2, కణుపు->src[3], కణుపు->src[4], కణుపు->src[5], కణుపు->src[6],
                                        కణుపు);
        స్థితి GGML_OP_GATED_DELTA_NET:
            ఒకవేళ (ggml_webgpu_can_fuse_gdn_cache(cgraph, కణుపు_idx, అంకె_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[కణుపు_idx + అంకె_encoded_ops - 1]->src[1];
                తిరిగి ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, కణుపు->src[3], కణుపు->src[4], కణుపు->src[5],
                                                   కణుపు, dst_fuse);
            } లేకపోతే {
                తిరిగి ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, కణుపు->src[3], కణుపు->src[4], కణుపు->src[5],
                                                   కణుపు, శూన్యసూచిక);
            }
        స్థితి GGML_OP_PAD:
            తిరిగి ggml_webgpu_pad(ctx, src0, కణుపు);
        స్థితి GGML_OP_ARGMAX:
            తిరిగి ggml_webgpu_argmax(ctx, src0, కణుపు);
        స్థితి GGML_OP_ARGSORT:
        స్థితి GGML_OP_శిఖరం_K:
            // we reuse the same argsort implementation for top_k
            తిరిగి ggml_webgpu_argsort(ctx, src0, కణుపు);
        స్థితి GGML_OP_CUMSUM:
            తిరిగి ggml_webgpu_cumsum(ctx, src0, కణుపు);
        స్థితి GGML_OP_కూడిక:
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            తిరిగి ggml_webgpu_కూడిక_అడ్డువరుసలు(ctx, src0, కణుపు);
        స్థితి GGML_OP_CONV_2D:
            తిరిగి ggml_webgpu_conv_2d(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_CONV_2D_DW:
            తిరిగి ggml_webgpu_conv_2d_dw(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_IM2_నిలువు:
            తిరిగి ggml_webgpu_im2col(ctx, src0, src1, కణుపు);
        స్థితి GGML_OP_UPSCALE:
            తిరిగి ggml_webgpu_upscale(ctx, src0, కణుపు);
        అప్రమేయ:
            తిరిగి std::nullopt;
    }
}

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
స్థిర శూన్య ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        స్థిరరాశి std::vector<std::string> & pipeline_పేర్లు,
                                                        uint32_t &                       అంకె_inflight_batches) {
    ఒకవేళ (pipeline_పేర్లు.ఖాళీలు()) {
        తిరిగి;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Resolve_Query_సమితి(ctx->profile_timestamp_query_సమితి, 0, ctx->profile_timestamp_query_లెక్క,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_బఫరు_గమ్యం_బఫరు(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_లెక్క * పరిమాణం(uint64_t));

    wgpu::Command_బఫరు profile_commands = encoder.ముగింపు_2();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, అంకె_inflight_batches);

    స్థిరరాశి size_t mapped_కొలత = ctx->profile_timestamp_query_లెక్క * పరిమాణం(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_లెక్క == 2 * pipeline_పేర్లు.size());

    ggml_backend_webgpu_map_బఫరు(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::Map_బహుళకం::Read, 0,
                                   mapped_కొలత);
    స్థిరరాశి uint64_t * ts_దత్తాంశం = (స్థిరరాశి uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_కొలత);

    కొరకు (size_t i = 0; i < pipeline_పేర్లు.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        స్థిరరాశి ద్విగుణ elapsed_ms = ద్విగుణ(ts_దత్తాంశం[2 * i + 1] - ts_దత్తాంశం[2 * i]) * 1e-6;
        ctx->shader_gpu_సమయం_ms[pipeline_పేర్లు[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#అయితే_ముగింపు

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
స్థిర శూన్య ggml_backend_webgpu_check_సమితి_అడ్డువరుసలు(webgpu_context & ctx, uint32_t & అంకె_inflight_batches) {
#నిర్వచితమైతే GGML_WEBGPU_CHECK_సమితి_అడ్డువరుసలు
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_బఫరు_గమ్యం_బఫరు(ctx->సమితి_అడ్డువరుసలు_dev_error_buf, 0, ctx->సమితి_అడ్డువరుసలు_host_error_buf, 0,
                               ctx->సమితి_అడ్డువరుసలు_host_error_buf.Get_కొలత());
    wgpu::Command_బఫరు commands = encoder.ముగింపు_2();
    ggml_backend_webgpu_submit_commands(ctx, commands, అంకె_inflight_batches);
    ggml_backend_webgpu_map_బఫరు(ctx->global_ctx, ctx->సమితి_అడ్డువరుసలు_host_error_buf, wgpu::Map_బహుళకం::Read, 0,
                                   ctx->సమితి_అడ్డువరుసలు_host_error_buf.Get_కొలత());
    స్థిరరాశి uint32_t * error_దత్తాంశం = (స్థిరరాశి uint32_t *) ctx->సమితి_అడ్డువరుసలు_host_error_buf.GetConstMappedRange();
    ఒకవేళ (*error_దత్తాంశం) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->సమితి_అడ్డువరుసలు_host_error_buf.Unmap();
#లేకపోతే
    GGML_వాడని(ctx);
    GGML_వాడని(అంకె_inflight_batches);
#అయితే_ముగింపు
}

స్థిర ggml_స్థితి ggml_backend_webgpu_లేఖాచిత్రం_compute(ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(లేఖాచిత్రం_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t అంకె_batched_kernels  = 0;
    uint32_t అంకె_inflight_batches = 0;
    తార్కిక     కలిగియుంది_సమితి_అడ్డువరుసలు    = అబద్ధం;
    పూర్ణాంకం      అంకె_encoded_ops      = 1;
    పూర్ణాంకం      కణుపు_idx             = 0;

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_లెక్క = 0;
    std::vector<std::string> profile_pipeline_పేర్లు;
#అయితే_ముగింపు

    ctx->చురుకైన_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    ఒకవేళ (ctx->batch_compute_passes) {
        ctx->చురుకైన_compute_pass = ctx->చురుకైన_command_encoder.BeginComputePass();
    }

    అయితే (కణుపు_idx < cgraph->n_nodes) {
        ఒకవేళ (cgraph->nodes[కణుపు_idx]->op == GGML_OP_సమితి_అడ్డువరుసలు) {
            కలిగియుంది_సమితి_అడ్డువరుసలు = నిజం;
        }
        ఒకవేళ (స్వయంచాలక cmd = ggml_webgpu_encode(ctx, cgraph, కణుపు_idx, అంకె_encoded_ops)) {
            commands.push_back(*cmd);
            అంకె_batched_kernels += cmd.value().అంకె_kernels;
#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_పేర్లు.insert(profile_pipeline_పేర్లు.end(), cmd->pipeline_పేర్లు.begin(),
                                          cmd->pipeline_పేర్లు.end());
#అయితే_ముగింపు
        }

        ఒకవేళ (అంకె_batched_kernels >= ctx->global_ctx->command_submit_batch_కొలత) {
            ఒకవేళ (ctx->చురుకైన_compute_pass) {
                ctx->చురుకైన_compute_pass.చివర_2();
            }
            అంకె_batched_kernels                = 0;
            wgpu::Command_బఫరు batch_commands = ctx->చురుకైన_command_encoder.ముగింపు_2();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, అంకె_inflight_batches);

            // reset state for next batch
            ctx->చురుకైన_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            ఒకవేళ (ctx->batch_compute_passes) {
                ctx->చురుకైన_compute_pass = ctx->చురుకైన_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            ఒకవేళ (ctx->profile_timestamp_query_లెక్క + 2 * ctx->global_ctx->command_submit_batch_కొలత >=
                WEBGPU_MAX_PROFILE_QUERY_లెక్క) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_పేర్లు, అంకె_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_లెక్క = 0;
                profile_pipeline_పేర్లు.clear();
            }
#అయితే_ముగింపు
        }

        కణుపు_idx += అంకె_encoded_ops;
        అంకె_encoded_ops = 1;
    }

    ఒకవేళ (ctx->చురుకైన_compute_pass) {
        ctx->చురుకైన_compute_pass.చివర_2();
        ctx->చురుకైన_compute_pass = శూన్యసూచిక;
    }

    ఒకవేళ (అంకె_batched_kernels > 0) {
        wgpu::Command_బఫరు batch_commands = ctx->చురుకైన_command_encoder.ముగింపు_2();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, అంకె_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->చురుకైన_command_encoder = శూన్యసూచిక;

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_పేర్లు, అంకె_inflight_batches);
#అయితే_ముగింపు

    ఒకవేళ (కలిగియుంది_సమితి_అడ్డువరుసలు) {
        ggml_backend_webgpu_check_సమితి_అడ్డువరుసలు(ctx, అంకె_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_మొత్తం_చివర(లేఖాచిత్రం_compute, ctx->global_ctx);
    తిరిగి GGML_స్థితి_SUCCESS;
}

నిర్మాణం ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    తార్కిక                  recorded = అబద్ధం;
};

స్థిర ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    స్వయంచాలక * event_ctx      = కొత్త ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    స్వయంచాలక * event   = కొత్త ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    తిరిగి event;
}

స్థిర శూన్య ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_వాడని(dev);
    తొలగించు static_cast<ggml_backend_webgpu_event_context *>(event->context);
    తొలగించు event;
}

స్థిర శూన్య ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_వాడని(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    ఒకవేళ (!event_ctx->recorded) {
        తిరిగి;
    }
    wgpu::Wait_స్థితి స్థితి_2 =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    ఒకవేళ (స్థితి_2 == wgpu::Wait_స్థితి::Timed_బహిర్గతం) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = అబద్ధం;
}

స్థిర శూన్య ggml_backend_webgpu_event_రికార్డు(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.On_Submitted_Work_పూర్తి(
        wgpu::Callback_బహుళకం::AllowSpontaneous, [](wgpu::వరుస_Work_పూర్తి_స్థితి, wgpu::StringView) {});
    event_ctx->recorded = నిజం;
}

స్థిర శూన్య ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_వాడని(backend);
    ggml_backend_webgpu_device_event_synchronize(శూన్యసూచిక, event);
}

స్థిర శూన్య ggml_backend_webgpu_సమితి_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 స్థిరరాశి శూన్య *   దత్తాంశం,
                                                 size_t         స్థానభ్రంశం,
                                                 size_t         size) {
    GGML_వాడని(backend);
    స్వయంచాలక * buf_ctx      = (ggml_backend_webgpu_బఫరు_context *) tensor->బఫరు->context;
    size_t మొత్తం_స్థానభ్రంశం = ggml_webgpu_tensor_స్థానభ్రంశం(tensor) + స్థానభ్రంశం;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_బఫరు(buf_ctx->బఫరు, మొత్తం_స్థానభ్రంశం, దత్తాంశం, (size / 4) * 4);

    ఒకవేళ (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t మిగిలిన_కొలత = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        కొరకు (size_t i = 0; i < మిగిలిన_కొలత; i++) {
            ((uint8_t *) &val32)[i] = ((స్థిరరాశి uint8_t *) దత్తాంశం)[size - మిగిలిన_కొలత + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_బఫరు_memset(buf_ctx->global_ctx, buf_ctx->బఫరు, val32,
                                          మొత్తం_స్థానభ్రంశం + (size - మిగిలిన_కొలత), మిగిలిన_కొలత);
    }
}

స్థిర శూన్య ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_వరుస(backend_ctx->webgpu_ctx->global_ctx);
}

స్థిర ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_సమితి_tensor_async,
    /* .get_tensor_async        = */ ఖాళీ,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ఖాళీ,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_webgpu_లేఖాచిత్రం_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_రికార్డు,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ ఖాళీ,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

స్థిర శూన్య ggml_backend_webgpu_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_webgpu_బఫరు_context * ctx = static_cast<ggml_backend_webgpu_బఫరు_context *>(బఫరు->context);
    ఒకవేళ (ctx != శూన్యసూచిక && ctx->బఫరు != శూన్యసూచిక) {
        ctx->బఫరు.Destroy();
        తొలగించు ctx;
    }
}

// Returns the "fake" base pointer.
స్థిర శూన్య * ggml_backend_webgpu_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    GGML_వాడని(బఫరు);
    తిరిగి webgpu_ptr_పునాది;
}

స్థిర శూన్య ggml_backend_webgpu_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                స్థానభ్రంశం,
                                                     size_t                size) {
    ఒకవేళ (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        తిరిగి;
    }

    WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(memset_tensor);

    ggml_backend_webgpu_బఫరు_context * buf_ctx = (ggml_backend_webgpu_బఫరు_context *) బఫరు->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << స్థానభ్రంశం << ", " << size << ")");

    size_t మొత్తం_స్థానభ్రంశం = ggml_webgpu_tensor_స్థానభ్రంశం(tensor) + స్థానభ్రంశం;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_బఫరు_memset(buf_ctx->global_ctx, buf_ctx->బఫరు, val32, మొత్తం_స్థానభ్రంశం, size);
    WEBGPU_CPU_PROFILE_మొత్తం_చివర(memset_tensor, buf_ctx->global_ctx);
}

స్థిర శూన్య ggml_backend_webgpu_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                                  ggml_tensor *         tensor,
                                                  స్థిరరాశి శూన్య *          దత్తాంశం,
                                                  size_t                స్థానభ్రంశం,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(సమితి_tensor);
    ggml_backend_webgpu_బఫరు_context * buf_ctx = (ggml_backend_webgpu_బఫరు_context *) బఫరు->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << దత్తాంశం
                                                              << ", " << స్థానభ్రంశం << ", " << size << ")");

    size_t మొత్తం_స్థానభ్రంశం = ggml_webgpu_tensor_స్థానభ్రంశం(tensor) + స్థానభ్రంశం;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    ఒకవేళ (మొత్తం_స్థానభ్రంశం % 4 != 0) {
        size_t lane = మొత్తం_స్థానభ్రంశం % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t తల = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t తల_val = 0;
        కొరకు (size_t i = 0; i < తల; i++) {
            ((uint8_t *) &తల_val)[lane + i] = ((స్థిరరాశి uint8_t *) దత్తాంశం)[i];
        }
        ggml_backend_webgpu_బఫరు_memset(buf_ctx->global_ctx, buf_ctx->బఫరు, తల_val, మొత్తం_స్థానభ్రంశం, తల);

        మొత్తం_స్థానభ్రంశం += తల;
        size -= తల;
        దత్తాంశం = (స్థిరరాశి uint8_t *) దత్తాంశం + తల;
    }

    ఒకవేళ (size > 0) {
        buf_ctx->global_ctx->queue.Write_బఫరు(buf_ctx->బఫరు, మొత్తం_స్థానభ్రంశం, దత్తాంశం, (size / 4) * 4);
    }

    ఒకవేళ (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t మిగిలిన_కొలత = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        కొరకు (size_t i = 0; i < మిగిలిన_కొలత; i++) {
            ((uint8_t *) &val32)[i] = ((స్థిరరాశి uint8_t *) దత్తాంశం)[size - మిగిలిన_కొలత + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_బఫరు_memset(buf_ctx->global_ctx, buf_ctx->బఫరు, val32,
                                          మొత్తం_స్థానభ్రంశం + (size - మిగిలిన_కొలత), మిగిలిన_కొలత);
    }
    WEBGPU_CPU_PROFILE_మొత్తం_చివర(సమితి_tensor, buf_ctx->global_ctx);
}

స్థిర శూన్య ggml_backend_webgpu_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                                  స్థిరరాశి ggml_tensor *   tensor,
                                                  శూన్య *                దత్తాంశం,
                                                  size_t                స్థానభ్రంశం,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(get_tensor);
    ggml_backend_webgpu_బఫరు_context * buf_ctx = (ggml_backend_webgpu_బఫరు_context *) బఫరు->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << దత్తాంశం
                                                              << ", " << స్థానభ్రంశం << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t మొత్తం_స్థానభ్రంశం = ggml_webgpu_tensor_స్థానభ్రంశం(tensor) + స్థానభ్రంశం;

    size_t local_స్థానభ్రంశం = మొత్తం_స్థానభ్రంశం % 4;
    ఒకవేళ (local_స్థానభ్రంశం != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        మొత్తం_స్థానభ్రంశం -= local_స్థానభ్రంశం;
    }

    size_t final_కొలత = size + local_స్థానభ్రంశం;
    ఒకవేళ (final_కొలత % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_కొలత += 4 - (final_కొలత % 4);
    }

    std::lock_guard<std::recursive_mutex> తాళం(buf_ctx->global_ctx->mutex);

    ఒకవేళ (buf_ctx->global_ctx->get_tensor_staging_buf == శూన్యసూచిక ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_కొలత() < final_కొలత) {
        // Create a new staging buffer if it doesn't exist or is too small
        ఒకవేళ (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_బఫరు(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_కొలత,
                                  wgpu::బఫరు_Usage::CopyDst | wgpu::బఫరు_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_బఫరు_గమ్యం_బఫరు(buf_ctx->బఫరు, మొత్తం_స్థానభ్రంశం, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_కొలత);
    wgpu::Command_బఫరు commands = encoder.ముగింపు_2();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_బఫరు(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::Map_బహుళకం::Read, 0, final_కొలత);
    // Must specify size here since the staging buffer might be larger than the tensor size
    స్థిరరాశి శూన్య * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_కొలత);

    // Copy the data from the mapped range to the output buffer
    std::జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి శూన్య *) ((స్థిరరాశి అక్షరం *) mapped_range + local_స్థానభ్రంశం), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_మొత్తం_చివర(get_tensor, buf_ctx->global_ctx);
}

స్థిర శూన్య ggml_backend_webgpu_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << బఫరు << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(clear);
    ggml_backend_webgpu_బఫరు_context * buf_ctx = (ggml_backend_webgpu_బఫరు_context *) బఫరు->context;
    ggml_backend_webgpu_బఫరు_memset(buf_ctx->global_ctx, buf_ctx->బఫరు, value, 0, బఫరు->size);
    WEBGPU_CPU_PROFILE_మొత్తం_చివర(clear, buf_ctx->global_ctx);
}

స్థిర ggml_backend_బఫరు_i ggml_backend_webgpu_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_webgpu_బఫరు_get_పునాది,
    /* .init_tensor     = */ ఖాళీ,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ఖాళీ,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_బఫరు_clear,
    /* .reset           = */ ఖాళీ,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

స్థిర స్థిరరాశి అక్షరం * ggml_backend_webgpu_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    తిరిగి ctx->device_name.c_str();
}

స్థిర ggml_backend_బఫరు_t ggml_backend_webgpu_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft,
                                                                          size_t                     size) {
    స్థిర std::atomic<పూర్ణాంకం> బఫరు_లెక్క;
    పూర్ణాంకం                     బఫరు_id = బఫరు_లెక్క++;
    std::string             buf_name  = "tensor_buf" + std::to_string(బఫరు_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << బఫరు_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::బఫరు_2                         buf;
    ggml_webgpu_create_బఫరు(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::బఫరు_Usage::Storage | wgpu::బఫరు_Usage::CopySrc | wgpu::బఫరు_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_బఫరు_context * buf_ctx =
        కొత్త ggml_backend_webgpu_బఫరు_context(buf, buf_name, ctx->webgpu_global_ctx);

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_webgpu_బఫరు_interface, buf_ctx, size);
}

స్థిర size_t ggml_backend_webgpu_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    తిరిగి dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
స్థిర size_t ggml_backend_webgpu_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    తిరిగి dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత;
}

స్థిర size_t ggml_backend_webgpu_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft,
                                                             స్థిరరాశి ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    ఎంపిక (tensor->op) {
        స్థితి GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            ఆపు;
        స్థితి GGML_OP_శిఖరం_K:
            {
                స్థిరరాశి ggml_tensor * src0 = tensor->src[0];
                ఒకవేళ (src0) {
                    స్థిరరాశి size_t నిండిన = పరిమాణం(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        నిండిన * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
                స్థిరరాశి ggml_tensor * Q            = tensor->src[0];
                స్థిరరాశి ggml_tensor * K            = tensor->src[1];
                స్థిరరాశి ggml_tensor * V            = tensor->src[2];
                స్థిరరాశి ggml_tensor * ముసుగు         = tensor->src[3];
                స్థిరరాశి స్వయంచాలక &        capabilities = ctx->webgpu_global_ctx->capabilities;
                ఒకవేళ (ggml_webgpu_flash_attn_use_vec_మార్గం(ctx->webgpu_global_ctx, Q, K, V)) {
                    స్థిరరాశి తార్కిక kv_ప్రత్యక్ష =
                        ggml_webgpu_flash_attn_k_ప్రత్యక్ష(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు) ||
                        ggml_webgpu_flash_attn_v_ప్రత్యక్ష(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు);
                    స్థిరరాశి uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_కొలత, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        ముసుగు != శూన్యసూచిక, kv_ప్రత్యక్ష);

                    స్థిరరాశి uint32_t vec_nwg_cap = capabilities.min_subgroup_కొలత;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    స్థిరరాశి size_t   align = capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;
                    స్థిరరాశి uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    ఒకవేళ (nwg > 1u) {
                        స్థిరరాశి uint64_t tmp_దత్తాంశం_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        స్థిరరాశి uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        స్థిరరాశి size_t   tmp_కొలత_బైట్లు = ROUNDUP_POW2((tmp_దత్తాంశం_elems + tmp_stats_elems) * పరిమాణం(దశాంశం),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_కొలత_బైట్లు + align;
                    } లేకపోతే {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    ఒకవేళ (ముసుగు != శూన్యసూచిక) {
                        స్థిరరాశి uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        స్థిరరాశి uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        స్థిరరాశి uint32_t stride_mask3    = (uint32_t) (ముసుగు->nb[3] / ggml_type_కొలత(ముసుగు->type));
                        స్థిరరాశి uint32_t blk_batch_లెక్క = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        స్థిరరాశి uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_లెక్క;
                        స్థిరరాశి size_t   blk_కొలత_బైట్లు =
                            ROUNDUP_POW2(blk_elems * పరిమాణం(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_కొలత_బైట్లు + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            {
                స్థిరరాశి ggml_tensor * src0 = tensor->src[0];
                స్థిరరాశి ggml_tensor * src1 = tensor->src[1];
                తార్కిక                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_లబ్ధం,
                                             ctx->webgpu_global_ctx->vendor);
                ఒకవేళ (use_mmvq) {
                    స్థిరరాశి size_t q8_src1_కొలత = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_కొలత +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ఆపు;
        స్థితి GGML_OP_MUL_MAT_ID:
            {
                స్థిరరాశి ggml_tensor * src0 = tensor->src[0];
                స్థిరరాశి ggml_tensor * src1 = tensor->src[1];
                ఒకవేళ (src0 && src1) {
                    స్థిరరాశి size_t ids_cache_కొలత = పరిమాణం(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    స్థిరరాశి size_t ids_tiles_కొలత = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_కొలత + ids_tiles_కొలత +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            ఆపు;
        అప్రమేయ:
            ఆపు;
    }
    తిరిగి res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

స్థిర స్థిరరాశి అక్షరం * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    తిరిగి ctx->device_name.c_str();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    తిరిగి ctx->device_desc.c_str();
}

స్థిర శూన్య ggml_backend_webgpu_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_బఫరు_కొలత = ctx->webgpu_global_ctx->capabilities.limits.max_బఫరు_కొలత_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#ఒకవేళ UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_కొలత = static_cast<uint64_t>(UINTPTR_MAX);
    ఒకవేళ (max_బఫరు_కొలత > max_ptr_కొలత) {
        max_బఫరు_కొలత = max_ptr_కొలత;
    }
#అయితే_ముగింపు
    *విడుదల  = static_cast<size_t>(max_బఫరు_కొలత);
    *మొత్తం = static_cast<size_t>(max_బఫరు_కొలత);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_వాడని(dev);
    తిరిగి GGML_BACKEND_DEVICE_TYPE_GPU;
}

స్థిర శూన్య ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                 = */ అబద్ధం,
        /* .host_buffer           = */ అబద్ధం,
        /* .buffer_from_host_ptr  = */ అబద్ధం,
        /* .events                = */ అబద్ధం,
        /* .mmap_support          = */ నిజం,
    };
}

స్థిర ggml_guid_t ggml_backend_webgpu_guid(శూన్య) {
    స్థిర ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    తిరిగి &guid;
}

స్థిర శూన్య ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_దారాలు = ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup *
                         ctx->capabilities.limits.max_Compute_Workgroups_చొప్పున_Dimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_బైట్లు_చొప్పున_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత, max_దారాలు);
    std::vector<wgpu::Constant_నమోదు> constants(2);
    constants[0].కీలకం     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
    constants[1].కీలకం     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_బైట్లు_చొప్పున_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

స్థిర శూన్య ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#ఒకవేళ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    స్థిరరాశి అక్షరం * స్థిరరాశి          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_లెక్క = 2;
    options.తదుపరి_లోపల_Chain                   = &adapterTogglesDesc;
#అయితే_ముగింపు

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::Callback_బహుళకం::AllowSpontaneous,
            [&adapter](wgpu::Request_Adapter_స్థితి స్థితి_2, wgpu::Adapter _adapter, wgpu::StringView సందేశం) {
                ఒకవేళ (స్థితి_2 != wgpu::Request_Adapter_స్థితి::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(సందేశం).c_str());
                    తిరిగి;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

స్థిర శూన్య create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != శూన్యసూచిక);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#ఒకవేళ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::Adapter_Properties_Subgroup_మాతృక_Configs subgroup_మాతృక_configs{};
    ఒకవేళ (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_మాతృక)) {
        info.తదుపరి_లోపల_Chain = &subgroup_మాతృక_configs;
    }
#అయితే_ముగింపు
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_కొలత = ggml_backend_webgpu_get_command_submit_batch_కొలత();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_లబ్ధం = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8_Integer_Dot_లబ్ధం);

    తార్కిక valid_subgroup_మాతృక_config = అబద్ధం;
#ఒకవేళ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    ఒకవేళ (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_మాతృక)) {
        కొరకు (size_t i = 0; i < subgroup_మాతృక_configs.config_లెక్క; i++) {
            స్థిరరాశి wgpu::Subgroup_మాతృక_Config config = subgroup_మాతృక_configs.configs[i];
            ఒకవేళ (config.componentType == wgpu::Subgroup_మాతృక_Component_Type::F16 &&
                config.ఫలితం_Component_Type == wgpu::Subgroup_మాతృక_Component_Type::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_మాతృక_config                  = నిజం;
                ఆపు;
            }
        }
    }
#అయితే_ముగింపు
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_మాతృక = valid_subgroup_మాతృక_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_కొలత = info.subgroup_Min_కొలత;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_కొలత = info.subgroup_Max_కొలత;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#ఒకవేళ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    ఒకవేళ (ctx->webgpu_global_ctx->capabilities.supports_subgroup_మాతృక) {
        required_features.push_back(wgpu::FeatureName::Chromium_Experimental_Subgroup_మాతృక);
    }
#అయితే_ముగింపు

    ఒకవేళ (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#అయితే_ముగింపు

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.దత్తాంశం();
    dev_desc.required_Feature_లెక్క = required_features.size();
    dev_desc.సమితి_Device_కోల్పోయిన_Callback(
        wgpu::Callback_బహుళకం::AllowSpontaneous,
        [](స్థిరరాశి wgpu::Device & device, wgpu::Device_కోల్పోయిన_Reason reason, wgpu::StringView సందేశం) {
            ఒకవేళ (reason == wgpu::Device_కోల్పోయిన_Reason::Destroyed) {
                తిరిగి;
            }
            GGML_వాడని(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<పూర్ణాంకం>(reason),
                           std::string(సందేశం).c_str());
        });
    dev_desc.సమితి_Uncaptured_Error_Callback(
        [](స్థిరరాశి wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView సందేశం) {
            GGML_వాడని(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<పూర్ణాంకం>(reason),
                       std::string(సందేశం).c_str());
        });

#ఒకవేళ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    స్థిరరాశి అక్షరం * స్థిరరాశి          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    స్థిరరాశి అక్షరం * స్థిరరాశి          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_లెక్క  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_లెక్క = 1;

    dev_desc.తదుపరి_లోపల_Chain = &deviceTogglesDesc;
#అయితే_ముగింపు

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::Callback_బహుళకం::AllowSpontaneous,
            [ctx](wgpu::Request_Device_స్థితి స్థితి_2, wgpu::Device device, wgpu::StringView సందేశం) {
                ఒకవేళ (స్థితి_2 != wgpu::Request_Device_స్థితి::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(సందేశం).c_str());
                    తిరిగి;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != శూన్యసూచిక);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_బఫరు(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_కొలత_బైట్లు, wgpu::బఫరు_Usage::CopyDst | wgpu::బఫరు_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.Get_వరుస();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

స్థిర webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_కొలత_బైట్లు,
        webgpu_ctx->global_ctx->command_submit_batch_కొలత + WEBGPU_అంకె_PARAM_స్థానం_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_ఏకరీతి_బఫరు_స్థానభ్రంశం_Alignment);
    ggml_webgpu_create_బఫరు(webgpu_ctx->global_ctx->device, webgpu_ctx->సమితి_అడ్డువరుసలు_dev_error_buf,
                              WEBGPU_సమితి_అడ్డువరుసలు_ERROR_BUF_కొలత_బైట్లు,
                              wgpu::బఫరు_Usage::Storage | wgpu::బఫరు_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_బఫరు(webgpu_ctx->global_ctx->device, webgpu_ctx->సమితి_అడ్డువరుసలు_host_error_buf,
                              WEBGPU_సమితి_అడ్డువరుసలు_ERROR_BUF_కొలత_బైట్లు,
                              wgpu::బఫరు_Usage::CopyDst | wgpu::బఫరు_Usage::MapRead, "set_rows_host_error_buf");

#నిర్వచితమైతే GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = అబద్ధం;
    ggml_webgpu_create_బఫరు(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_కొలత_బైట్లు,
        wgpu::బఫరు_Usage::QueryResolve | wgpu::బఫరు_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_బఫరు(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_కొలత_బైట్లు,
                              wgpu::బఫరు_Usage::CopyDst | wgpu::బఫరు_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::Query_సమితి_Descriptor query_సమితి_desc = {};
    query_సమితి_desc.type                     = wgpu::QueryType::Timestamp;
    query_సమితి_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_లెక్క;
    webgpu_ctx->profile_timestamp_query_సమితి = webgpu_ctx->global_ctx->device.Create_Query_సమితి(&query_సమితి_desc);
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_బఫరు(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * పరిమాణం(uint32_t),
                              wgpu::బఫరు_Usage::CopyDst | wgpu::బఫరు_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_బఫరు(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * పరిమాణం(uint32_t),
                              wgpu::బఫరు_Usage::Storage | wgpu::బఫరు_Usage::CopySrc, "debug_dev_buf");
#అయితే_ముగింపు
    తిరిగి webgpu_ctx;
}

స్థిర ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    GGML_వాడని(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    స్వయంచాలక * backend_ctx      = కొత్త ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    స్వయంచాలక * backend = కొత్త ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    తిరిగి backend;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_webgpu_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_webgpu_బఫరు_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_బఫరు_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_బఫరు_type_alloc_బఫరు,
                        /* .alloc_buffer_n      = */ ఖాళీ,
                        /* .get_alignment       = */ ggml_backend_webgpu_బఫరు_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_బఫరు_type_get_max_కొలత,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_బఫరు_type_get_alloc_కొలత,
                        /* .get_alloc_size_n    = */ ఖాళీ,
                        /* .is_host             = */ ఖాళీ,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ ఖాళీ
    };

    తిరిగి &ggml_backend_webgpu_బఫరు_type;
}

స్థిర తార్కిక ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    GGML_వాడని(dev);
    తిరిగి buft->iface.get_name == ggml_backend_webgpu_బఫరు_type_get_name;
}

స్థిర తార్కిక ggml_webgpu_supported_qtype(ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

స్థిర తార్కిక ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    ఒకవేళ (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత ||
        (src0 != శూన్యసూచిక &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత) ||
        (src1 != శూన్యసూచిక &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత)) {
        తిరిగి అబద్ధం;
    }

    తార్కిక supports_op = అబద్ధం;
    ఎంపిక (op->op) {
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_RESHAPE:
            supports_op = నిజం;
            ఆపు;
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            ఆపు;
        స్థితి GGML_OP_కూడు_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            ఆపు;
        స్థితి GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            ఆపు;
        స్థితి GGML_OP_DUP:
        స్థితి GGML_OP_CPY:
        స్థితి GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            ఆపు;
        స్థితి GGML_OP_సమితి:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            ఆపు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            ఒకవేళ (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            {
                ఎంపిక (src1->type) {
                    స్థితి GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        ఆపు;
                    స్థితి GGML_TYPE_F32:
                        ఎంపిక (src0->type) {
                            స్థితి GGML_TYPE_F32:
                            స్థితి GGML_TYPE_F16:
                            స్థితి GGML_TYPE_BF16:
                            స్థితి GGML_TYPE_Q1_0:
                            స్థితి GGML_TYPE_Q2_0:
                            స్థితి GGML_TYPE_Q4_0:
                            స్థితి GGML_TYPE_Q4_1:
                            స్థితి GGML_TYPE_Q5_0:
                            స్థితి GGML_TYPE_Q5_1:
                            స్థితి GGML_TYPE_Q8_0:
                            స్థితి GGML_TYPE_Q2_K:
                            స్థితి GGML_TYPE_Q3_K:
                            స్థితి GGML_TYPE_Q4_K:
                            స్థితి GGML_TYPE_Q5_K:
                            స్థితి GGML_TYPE_Q6_K:
                            స్థితి GGML_TYPE_IQ2_XXS:
                            స్థితి GGML_TYPE_IQ2_XS:
                            స్థితి GGML_TYPE_IQ2_S:
                            స్థితి GGML_TYPE_IQ3_XXS:
                            స్థితి GGML_TYPE_IQ3_S:
                            స్థితి GGML_TYPE_IQ1_S:
                            స్థితి GGML_TYPE_IQ1_M:
                            స్థితి GGML_TYPE_IQ4_NL:
                            స్థితి GGML_TYPE_IQ4_XS:
                            స్థితి GGML_TYPE_MXFP4:
                            స్థితి GGML_TYPE_NVFP4:
                                supports_op = నిజం;
                                ఆపు;
                            అప్రమేయ:
                                ఆపు;
                        }
                    అప్రమేయ:
                        ఆపు;
                }
                ఆపు;
            }
        స్థితి GGML_OP_MUL_MAT_ID:
            ఎంపిక (src1->type) {
                స్థితి GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    ఆపు;
                స్థితి GGML_TYPE_F32:
                    ఎంపిక (src0->type) {
                        స్థితి GGML_TYPE_F32:
                        స్థితి GGML_TYPE_F16:
                        స్థితి GGML_TYPE_BF16:
                        స్థితి GGML_TYPE_Q1_0:
                        స్థితి GGML_TYPE_Q4_0:
                        స్థితి GGML_TYPE_Q4_1:
                        స్థితి GGML_TYPE_Q5_0:
                        స్థితి GGML_TYPE_Q5_1:
                        స్థితి GGML_TYPE_Q8_0:
                        స్థితి GGML_TYPE_Q2_K:
                        స్థితి GGML_TYPE_Q3_K:
                        స్థితి GGML_TYPE_Q4_K:
                        స్థితి GGML_TYPE_Q5_K:
                        స్థితి GGML_TYPE_Q6_K:
                        స్థితి GGML_TYPE_IQ1_S:
                        స్థితి GGML_TYPE_IQ1_M:
                        స్థితి GGML_TYPE_IQ2_XXS:
                        స్థితి GGML_TYPE_IQ2_XS:
                        స్థితి GGML_TYPE_IQ2_S:
                        స్థితి GGML_TYPE_IQ3_XXS:
                        స్థితి GGML_TYPE_IQ3_S:
                        స్థితి GGML_TYPE_IQ4_NL:
                        స్థితి GGML_TYPE_IQ4_XS:
                        స్థితి GGML_TYPE_MXFP4:
                        స్థితి GGML_TYPE_NVFP4:
                            supports_op = నిజం;
                            ఆపు;
                        అప్రమేయ:
                            ఆపు;
                    }
                    ఆపు;
                అప్రమేయ:
                    ఆపు;
            }
            ఒకవేళ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                ఒకవేళ (!supports_op) {
                    ఆపు;
                }
                ఒకవేళ (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = అబద్ధం;
                    ఆపు;
                }
                స్థిరరాశి స్వయంచాలక & capabilities             = ctx->webgpu_global_ctx->capabilities;
                స్థిరరాశి size_t storage_స్థానభ్రంశం_alignment = capabilities.limits.min_Storage_బఫరు_స్థానభ్రంశం_Alignment;

                // subgroup matrix path requirements
                స్థిరరాశి తార్కిక use_subgroup_మాతృక = ggml_webgpu_flash_attn_can_use_subgroup_మాతృక_మార్గం(
                    capabilities.supports_subgroup_మాతృక, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                స్థిరరాశి తార్కిక float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_స్థానభ్రంశం_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_స్థానభ్రంశం_alignment));
                స్థిరరాశి uint32_t k_tile_తల_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు :
                                                       (uint32_t) ggml_blck_కొలత(src1->type);
                స్థిరరాశి uint32_t v_tile_తల_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_వెడల్పు :
                                                       (uint32_t) ggml_blck_కొలత(src2->type);
                స్థిరరాశి తార్కిక     tile_kv_తల_dims_aligned =
                    src0->ne[0] % k_tile_తల_align == 0 && src2->ne[0] % v_tile_తల_align == 0;
                స్థిరరాశి తార్కిక tile_can_dispatch_all_q_అడ్డువరుసలు =
                    capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_కొలత;
                స్థిరరాశి తార్కిక use_tile = !use_subgroup_మాతృక && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_తల_dims_aligned && tile_can_dispatch_all_q_అడ్డువరుసలు;

                ఒకవేళ (!use_subgroup_మాతృక && !use_tile) {
                    supports_op = అబద్ధం;
                    ఆపు;
                }
                స్థిరరాశి uint32_t q_tile =
                    use_subgroup_మాతృక ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                స్థిరరాశి uint32_t kv_granularity = use_subgroup_మాతృక ? capabilities.sg_mat_n : 1u;
                స్థిరరాశి తార్కిక     kv_ప్రత్యక్ష = use_subgroup_మాతృక ?
                                               ggml_webgpu_flash_attn_k_ప్రత్యక్ష(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_ప్రత్యక్ష(src0, src2, capabilities.sg_mat_k) :
                                               అబద్ధం;
                స్థిరరాశి uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_కొలత, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != శూన్యసూచిక, kv_ప్రత్యక్ష);
                supports_op = max_kv_tile > 0;
                ఆపు;
            }
        స్థితి GGML_OP_RMS_NORM:
        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_అడ్డువరుసలు(src0);
            ఆపు;
        స్థితి GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            ఆపు;
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(op)) {
                స్థితి GGML_GLU_OP_REGLU:
                స్థితి GGML_GLU_OP_GEGLU:
                స్థితి GGML_GLU_OP_SWIGLU:
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    ఆపు;
                అప్రమేయ:
                    ఆపు;
            }
            ఆపు;
        స్థితి GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_UNARY:
            {
                స్థిరరాశి ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                ఎంపిక (UNARY_OP) {
                    స్థితి GGML_UNARY_OP_ABS:
                    స్థితి GGML_UNARY_OP_SGN:
                    స్థితి GGML_UNARY_OP_NEG:
                    స్థితి GGML_UNARY_OP_అడుగు:
                    స్థితి GGML_UNARY_OP_TANH:
                    స్థితి GGML_UNARY_OP_ELU:
                    స్థితి GGML_UNARY_OP_RELU:
                    స్థితి GGML_UNARY_OP_SIGMOID:
                    స్థితి GGML_UNARY_OP_GELU:
                    స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                    స్థితి GGML_UNARY_OP_SILU:
                    స్థితి GGML_UNARY_OP_HARDSWISH:
                    స్థితి GGML_UNARY_OP_HARDSIGMOID:
                    స్థితి GGML_UNARY_OP_EXP:
                    స్థితి GGML_UNARY_OP_GELU_ERF:
                    స్థితి GGML_UNARY_OP_SOFTPLUS:
                    స్థితి GGML_UNARY_OP_EXPM1:
                    స్థితి GGML_UNARY_OP_FLOOR:
                    స్థితి GGML_UNARY_OP_CEIL:
                    స్థితి GGML_UNARY_OP_విడత:
                    స్థితి GGML_UNARY_OP_TRUNC:
                    స్థితి GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        ఆపు;
                    అప్రమేయ:
                        ఆపు;
                }
            }
            ఆపు;
        స్థితి GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            ఆపు;
        స్థితి GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            ఆపు;
        స్థితి GGML_OP_IM2_నిలువు:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            ఆపు;
        స్థితి GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_SSM_పరిశీలన:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
            ఆపు;
        స్థితి GGML_OP_GATED_DELTA_NET:
            {
                స్థిరరాశి uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_చొప్పున_Workgroup;
            }
            ఆపు;
        స్థితి GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ఆపు;
        స్థితి GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ఆపు;
        స్థితి GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ఆపు;
        స్థితి GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ఆపు;
        స్థితి GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ఆపు;
        స్థితి GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ఆపు;
        స్థితి GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            ఆపు;
        స్థితి GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            ఆపు;
        స్థితి GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_అడ్డువరుసలు(src0);
            ఆపు;
        స్థితి GGML_OP_శిఖరం_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_అడ్డువరుసలు(src0);
            ఆపు;
        స్థితి GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            ఆపు;
        స్థితి GGML_OP_కూడిక:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_అడ్డువరుసలు(src0);
            ఆపు;
        స్థితి GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            ఆపు;
        అప్రమేయ:
            ఆపు;
    }
    ఒకవేళ (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత ||
        (src0 != శూన్యసూచిక &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత) ||
        (src1 != శూన్యసూచిక &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత) ||
        (src2 != శూన్యసూచిక &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_బఫరు_Binding_కొలత)) {
        supports_op = అబద్ధం;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    ఒకవేళ (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } లేకపోతే {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    తిరిగి supports_op;
}

స్థిర నిర్మాణం ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ఖాళీ,
    /* .buffer_from_host_ptr = */ ఖాళీ,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ ఖాళీ,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

స్థిర స్థిరరాశి అక్షరం * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    తిరిగి ctx->name;
}

స్థిర size_t ggml_backend_webgpu_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    తిరిగి ctx->device_లెక్క;
}

// Only one device is supported for now
స్థిర ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    GGML_ASSERT(సూచీ == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_మొత్తం_ప్రారంభం(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    స్థిర ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    స్థిర ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_మొత్తం_చివర(reg_get_device, reg_ctx->webgpu_global_ctx);
    తిరిగి &device;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ ఖాళీ,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    స్థిర ggml_backend_webgpu_reg_context * ctx = కొత్త ggml_backend_webgpu_reg_context();

    స్థిర ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_లెక్క = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    ఒకవేళ (ctx->webgpu_global_ctx != శూన్యసూచిక && ctx->webgpu_global_ctx->instance != శూన్యసూచిక) {
        తిరిగి &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.దత్తాంశం();
    instance_descriptor.required_Feature_లెక్క                 = instance_features.size();

#ఒకవేళ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    స్థిరరాశి అక్షరం * స్థిరరాశి          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_లెక్క = 1;
    instance_descriptor.తదుపరి_లోపల_Chain        = &instanceTogglesDesc;
#అయితే_ముగింపు

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(కొత్త webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    ఒకవేళ (ctx->webgpu_global_ctx->instance != శూన్యసూచిక) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    ఒకవేళ (adapter != శూన్యసూచిక && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#ఒకవేళ !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#అయితే_ముగింపు
    ) {
        ctx->device_లెక్క = 1;
    }

    తిరిగి &reg;
}

ggml_backend_t ggml_backend_webgpu_init(శూన్య) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    ఒకవేళ (ggml_backend_reg_dev_లెక్క(reg) == 0) {
        తిరిగి శూన్యసూచిక;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    తిరిగి ggml_backend_webgpu_backend_init(dev, శూన్యసూచిక);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
