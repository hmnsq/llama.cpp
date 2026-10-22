/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#अन्तर्भाव "ggml-webgpu.h"

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-webgpu-shader-lib.hpp"
#अन्तर्भाव "ggml.h"

#यदि_परिभाषितम् __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#यदि_अन्तः

#अन्तर्भाव <webgpu/webgpu_cpp.h>

#अन्तर्भाव <atomic>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstring>
#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#यदि_अन्तः
#यदि defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#यदि_अन्तः
#अन्तर्भाव <memory>
#अन्तर्भाव <mutex>
#अन्तर्भाव <optional>
#अन्तर्भाव <string>
#अन्तर्भाव <utility>
#अन्तर्भाव <vector>

#परिभाषय ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#परिभाषय CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
स्थिर अन्तःस्थ शून्य compute_2d_workgroups(uint32_t योग_wg, uint32_t max_प्रत्येकम्_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(योग_wg, max_प्रत्येकम्_dim));
    wg_x = CEIL_DIV(योग_wg, wg_y);
}

स्थिर अन्तःस्थ uint32_t ggml_webgpu_u32_from_f32(भिन्न value) {
    uint32_t द्वयङ्काः;
    स्मृतिं_प्रतिलिख(&द्वयङ्काः, &value, परिमाण(द्वयङ्काः));
    प्रत्यागम द्वयङ्काः;
}

#यदि_परिभाषितम् GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(सन्देशः_2)  std::निर्गमः << सन्देशः_2 << std::पङ्क्त्यन्तः
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#अन्यथा
#    define WEBGPU_LOG_DEBUG(सन्देशः_2) ((शून्य) 0)
#यदि_अन्तः  // GGML_WEBGPU_DEBUG

#यदि_परिभाषितम् GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_योग_START(id) स्वचालित cpu_योग_start##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_योग_अन्तः(id, ctx)                                                         \
        स्वचालित   cpu_योग_अन्तः##id = std::chrono::high_resolution_clock::now();                            \
        द्विगुण cpu_योग_time##id =                                                                      \
            std::chrono::duration<द्विगुण, std::milli>(cpu_योग_अन्तः##id - cpu_योग_start##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_योग_time##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) स्वचालित cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_अन्तः(id, ctx)                                                          \
        स्वचालित   cpu_detail_अन्तः##id = std::chrono::high_resolution_clock::now();                             \
        द्विगुण cpu_detail_time_##id =                                                                       \
            std::chrono::duration<द्विगुण, std::milli>(cpu_detail_अन्तः##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#अन्यथा
#    define WEBGPU_CPU_PROFILE_योग_START(id)
#    define WEBGPU_CPU_PROFILE_योग_अन्तः(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_अन्तः(id, ctx)
#यदि_अन्तः  // GGML_WEBGPU_CPU_PROFILE

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_गणनम्        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_आकार_अष्टकानि (WEBGPU_MAX_PROFILE_QUERY_गणनम् * परिमाण(uint64_t))
#यदि_अन्तः

/* Constants */

#परिभाषय WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_आकार 64u
#परिभाषय WEBGPU_अङ्कः_PARAM_SLOT_SAFETY_MARGIN      10u
#परिभाषय WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#परिभाषय WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#परिभाषय WEBGPU_PARAMS_BUF_आकार_अष्टकानि             128  // enough for 32 parameters
#परिभाषय WEBGPU_समुच्चयः_पङ्क्तिकाः_ERROR_BUF_आकार_अष्टकानि     4
#परिभाषय WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
स्थिर शून्य * ध्रुव webgpu_ptr_आधारः = (शून्य *) (uintptr_t) 0x1000;  // NOLINT

स्थिर size_t ggml_webgpu_tensor_विस्थापनम्(ध्रुव ggml_tensor * tensor) {
    ध्रुव ggml_tensor * आधारः_tensor = tensor->view_src ? tensor->view_src : tensor;
    प्रत्यागम (size_t) ((uintptr_t) आधारः_tensor->दत्त - (uintptr_t) webgpu_ptr_आधारः) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
स्थिर शून्य ggml_webgpu_create_बफर(wgpu::Device &    device,
                                      wgpu::बफर_2 &    बफर,
                                      size_t            size,
                                      wgpu::बफर_Usage usage,
                                      ध्रुव वर्ण *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
संरचना webgpu_param_arena {
    wgpu::बफर_2 बफर;
    size_t       slot_stride = 0;
    size_t       slot_आकार   = 0;
    uint32_t     slot_गणनम्  = 0;
    uint32_t     अग्र_slot   = 0;

    शून्य init(wgpu::Device device, size_t slot_आकार, uint32_t slot_गणनम्, size_t alignment) {
        एतत्->slot_stride = ROUNDUP_POW2(slot_आकार, alignment);
        एतत्->slot_आकार   = slot_आकार;
        एतत्->slot_गणनम्  = slot_गणनम्;
        एतत्->अग्र_slot   = 0;

        ggml_webgpu_create_बफर(device, बफर, एतत्->slot_stride * slot_गणनम्,
                                  wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_slot(size_t size) {
        GGML_ASSERT(size <= slot_आकार);
        यदि (अग्र_slot >= slot_गणनम्) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        प्रत्यागम slot_stride * अग्र_slot++;
    }

    शून्य reset() { अग्र_slot = 0; }

    शून्य cleanup() {
        यदि (बफर) {
            बफर.Destroy();
            बफर = शून्यसूचकः;
        }
    }

    ~webgpu_param_arena() { एतत्->cleanup(); }
};

संरचना webgpu_encoded_op {
    uint32_t अङ्कः_kernels = 0;
#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_नामानि;
#यदि_अन्तः
};

संरचना webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> bind_गणः_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

संरचना webgpu_capabilities {
    wgpu::Limits limits;
    तार्किक         supports_subgroups       = असत्य;
    तार्किक         supports_subgroup_matrix = असत्य;
    तार्किक         supports_dot_गुणनफलम्     = असत्य;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_आकार     = 0;
    uint32_t min_subgroup_आकार = 0;
    uint32_t max_subgroup_आकार = 0;
    size_t   memset_अष्टकानि_प्रत्येकम्_thread;
};

// Stores global webgpu members
संरचना webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    uint32_t       command_submit_batch_आकार = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_आकार;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::बफर_2         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::बफर_2    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#यदि_परिभाषितम् GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, द्विगुण> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, द्विगुण> cpu_detail_ms;
#यदि_अन्तः

#यदि_परिभाषितम् GGML_WEBGPU_DEBUG
    wgpu::बफर_2 debug_host_buf;
    wgpu::बफर_2 debug_dev_buf;
#यदि_अन्तः

    ~webgpu_global_context_struct() {
        यदि (एतत्->get_tensor_staging_buf) {
            एतत्->get_tensor_staging_buf.Destroy();
            एतत्->get_tensor_staging_buf = शून्यसूचकः;
        }
        यदि (एतत्->memset_params_buf) {
            एतत्->memset_params_buf.Destroy();
            एतत्->memset_params_buf = शून्यसूचकः;
        }
#यदि_परिभाषितम् GGML_WEBGPU_DEBUG
        यदि (एतत्->debug_host_buf) {
            एतत्->debug_host_buf.Destroy();
            एतत्->debug_host_buf = शून्यसूचकः;
        }
        यदि (एतत्->debug_dev_buf) {
            एतत्->debug_dev_buf.Destroy();
            एतत्->debug_dev_buf = शून्यसूचकः;
        }
#यदि_अन्तः
    }
};

प्रकारनिर्देश std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
संरचना webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::बफर_2             समुच्चयः_पङ्क्तिकाः_dev_error_buf;
    wgpu::बफर_2             समुच्चयः_पङ्क्तिकाः_host_error_buf;
    wgpu::CommandEncoder     सक्रिय_command_encoder;
    wgpu::ComputePassEncoder सक्रिय_compute_pass;
    तार्किक                     batch_compute_passes = सत्य;

    size_t memset_अष्टकानि_प्रत्येकम्_thread;

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, द्विगुण> shader_gpu_time_ms;
    wgpu::बफर_2                            profile_timestamp_dev_buf;
    wgpu::बफर_2                            profile_timestamp_host_buf;
    wgpu::Query_समुच्चयः                          profile_timestamp_query_समुच्चयः;
    uint32_t                                profile_timestamp_query_गणनम् = 0;
#यदि_अन्तः

    ~webgpu_context_struct() {
#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
        यदि (एतत्->profile_timestamp_host_buf) {
            एतत्->profile_timestamp_host_buf.Destroy();
            एतत्->profile_timestamp_host_buf = शून्यसूचकः;
        }
        यदि (एतत्->profile_timestamp_dev_buf) {
            एतत्->profile_timestamp_dev_buf.Destroy();
            एतत्->profile_timestamp_dev_buf = शून्यसूचकः;
        }
        यदि (एतत्->profile_timestamp_query_समुच्चयः) {
            एतत्->profile_timestamp_query_समुच्चयः.Destroy();
            एतत्->profile_timestamp_query_समुच्चयः = शून्यसूचकः;
        }
#यदि_अन्तः
        यदि (एतत्->समुच्चयः_पङ्क्तिकाः_host_error_buf) {
            एतत्->समुच्चयः_पङ्क्तिकाः_host_error_buf.Destroy();
            एतत्->समुच्चयः_पङ्क्तिकाः_host_error_buf = शून्यसूचकः;
        }
        यदि (एतत्->समुच्चयः_पङ्क्तिकाः_dev_error_buf) {
            एतत्->समुच्चयः_पङ्क्तिकाः_dev_error_buf.Destroy();
            एतत्->समुच्चयः_पङ्क्तिकाः_dev_error_buf = शून्यसूचकः;
        }
    }
};

प्रकारनिर्देश std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
संरचना ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_गणनम्;
    ध्रुव वर्ण *          name;
};

// Per-device struct for the global logical device interface
संरचना ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
संरचना ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
संरचना ggml_backend_webgpu_बफर_context {
    wgpu::बफर_2          बफर;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_बफर_context(wgpu::बफर_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        बफर(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

स्थिर webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ध्रुव वर्ण *                             shader_संहिता,
                                                   ध्रुव वर्ण *                             label,
                                                   ध्रुव std::vector<wgpu::Constant_प्रविष्टिः> & constants = {}) {
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_संहिता;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.अग्र_अन्तर्_Chain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.प्रविष्टिः_Point = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = शून्यसूचकः;  // nullptr means auto layout
    यदि (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.दत्त();
        pipeline_desc.compute.constant_गणनम् = constants.size();
    }
    प्रत्यागम { device.CreateComputePipeline(&pipeline_desc), label };
}

स्थिर शून्य ggml_webgpu_create_बफर(wgpu::Device &    device,
                                      wgpu::बफर_2 &    बफर,
                                      size_t            size,
                                      wgpu::बफर_Usage usage,
                                      ध्रुव वर्ण *      label) {
    wgpu::बफर_Descriptor बफर_desc;
    बफर_desc.size             = size;
    बफर_desc.usage            = usage;
    बफर_desc.label            = label;
    बफर_desc.mappedAtCreation = असत्य;

    // TODO: error handling
    बफर = device.Create_बफर(&बफर_desc);
}

स्थिर wgpu::बफर_2 ggml_webgpu_tensor_buf(ध्रुव ggml_tensor * tensor) {
    ggml_backend_webgpu_बफर_context * ctx = (ggml_backend_webgpu_बफर_context *) tensor->बफर->context;
    प्रत्यागम ctx->बफर;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
स्थिर size_t ggml_webgpu_tensor_align_विस्थापनम्(ध्रुव ggml_tensor * t, size_t alignment) {
    ध्रुव size_t विस्थापनम्    = ggml_webgpu_tensor_विस्थापनम्(t);
    ध्रुव size_t type_आकार = ggml_type_आकार(t->type);
    size_t       aligned   = विस्थापनम् & ~(alignment - 1);
    यावत् ((विस्थापनम् - aligned) % type_आकार != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    प्रत्यागम aligned;
}

स्थिर size_t ggml_webgpu_tensor_misalignment(ध्रुव ggml_tensor * t, size_t alignment) {
    प्रत्यागम ggml_webgpu_tensor_विस्थापनम्(t) - ggml_webgpu_tensor_align_विस्थापनम्(t, alignment);
}

स्थिर size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ध्रुव ggml_tensor * t) {
    प्रत्यागम ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment);
}

स्थिर size_t ggml_webgpu_tensor_align_विस्थापनम्(webgpu_context & ctx, ध्रुव ggml_tensor * t) {
    प्रत्यागम ggml_webgpu_tensor_align_विस्थापनम्(t, ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment);
}

स्थिर size_t ggml_webgpu_tensor_binding_आकार(ध्रुव ggml_tensor * t, size_t alignment) {
    प्रत्यागम ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

स्थिर size_t ggml_webgpu_tensor_binding_आकार(webgpu_context & ctx, ध्रुव ggml_tensor * t) {
    प्रत्यागम ggml_webgpu_tensor_binding_आकार(t, ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment);
}

स्थिर तार्किक ggml_webgpu_tensor_binding_overlap(ध्रुव webgpu_global_context & global_ctx,
                                               ध्रुव ggml_tensor *           a,
                                               ध्रुव ggml_tensor *           b) {
    यदि (a->बफर != b->बफर) {
        प्रत्यागम असत्य;
    }

    ध्रुव size_t alignment = global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment;
    ध्रुव size_t a_विस्थापनम्  = ggml_webgpu_tensor_align_विस्थापनम्(a, alignment);
    ध्रुव size_t b_विस्थापनम्  = ggml_webgpu_tensor_align_विस्थापनम्(b, alignment);
    प्रत्यागम a_विस्थापनम् < b_विस्थापनम् + ggml_webgpu_tensor_binding_आकार(b, alignment) &&
           b_विस्थापनम् < a_विस्थापनम् + ggml_webgpu_tensor_binding_आकार(a, alignment);
}

स्थिर तार्किक ggml_webgpu_tensor_binding_overlap_range(ध्रुव webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_बफर_t         बफर,
                                                     size_t                        विस्थापनम्,
                                                     size_t                        size) {
    यदि (tensor->बफर != बफर) {
        प्रत्यागम असत्य;
    }

    ध्रुव size_t alignment     = global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment;
    ध्रुव size_t tensor_विस्थापनम् = ggml_webgpu_tensor_align_विस्थापनम्(tensor, alignment);
    प्रत्यागम tensor_विस्थापनम् < विस्थापनम् + size && विस्थापनम् < tensor_विस्थापनम् + ggml_webgpu_tensor_binding_आकार(tensor, alignment);
}

संरचना ggml_webgpu_merged_binding_range {
    size_t विस्थापनम्;
    size_t size;
};

स्थिर ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_विस्थापनम् = SIZE_MAX;
    size_t merged_अन्तः    = 0;

    प्रति (ggml_tensor * tensor : tensors) {
        ध्रुव size_t bind_विस्थापनम् = ggml_webgpu_tensor_align_विस्थापनम्(ctx, tensor);
        ध्रुव size_t bind_अन्तः    = bind_विस्थापनम् + ggml_webgpu_tensor_binding_आकार(ctx, tensor);

        merged_विस्थापनम् = std::min(merged_विस्थापनम्, bind_विस्थापनम्);
        merged_अन्तः    = std::max(merged_अन्तः, bind_अन्तः);
    }

    प्रत्यागम { merged_विस्थापनम्, merged_अन्तः - merged_विस्थापनम् };
}

स्थिर uint32_t ggml_webgpu_tensor_merged_element_विस्थापनम्(ध्रुव ggml_tensor *                      tensor,
                                                         ध्रुव ggml_webgpu_merged_binding_range & merged_range) {
    प्रत्यागम (uint32_t) ((ggml_webgpu_tensor_विस्थापनम्(tensor) - merged_range.विस्थापनम्) / ggml_type_आकार(tensor->type));
}

स्थिर wgpu::Bind_गणः_प्रविष्टिः ggml_webgpu_make_bind_गणः_प्रविष्टिः(uint32_t     binding,
                                                              wgpu::बफर_2 बफर,
                                                              uint64_t     विस्थापनम्,
                                                              uint64_t     size) {
    wgpu::Bind_गणः_प्रविष्टिः प्रविष्टिः = {};
    प्रविष्टिः.binding              = binding;
    प्रविष्टिः.बफर               = std::move(बफर);
    प्रविष्टिः.विस्थापनम्               = विस्थापनम्;
    प्रविष्टिः.size                 = size;
    प्रत्यागम प्रविष्टिः;
}

स्थिर wgpu::Bind_गणः_प्रविष्टिः ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    प्रत्यागम ggml_webgpu_make_bind_गणः_प्रविष्टिः(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_विस्थापनम्(ctx, tensor),
                                             ggml_webgpu_tensor_binding_आकार(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य ggml_backend_webgpu_check_wait_स्थितिः(wgpu::Wait_स्थितिः wait_स्थितिः,
                                                  T                callback_स्थितिः,
                                                  T                success_स्थितिः,
                                                  ध्रुव वर्ण *     wait_name,
                                                  ध्रुव वर्ण *     failure_name,
                                                  ध्रुव वर्ण *     callback_सन्देशः) {
    यदि (wait_स्थितिः == wgpu::Wait_स्थितिः::Timed_बहिर्गमनम्) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    यदि (wait_स्थितिः == wgpu::Wait_स्थितिः::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    यदि (callback_स्थितिः != success_स्थितिः) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<पूर्णाङ्क>(callback_स्थितिः),
                   callback_सन्देशः);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
स्थिर uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    प्रत्यागम UINT32_MAX;
}

स्थिर uint32_t ggml_backend_webgpu_get_command_submit_batch_आकार() {
    प्रत्यागम WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_आकार;
}

स्थिर शून्य ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::Queue_Work_समाप्तम्_स्थितिः callback_स्थितिः = wgpu::Queue_Work_समाप्तम्_स्थितिः::Error;
    std::string               callback_सन्देशः;

    ध्रुव wgpu::Wait_स्थितिः wait_स्थितिः = ctx->instance.WaitAny(
        ctx->queue.On_Submitted_Work_समाप्तम्(
            wgpu::Callback_बहुलकम्::AllowSpontaneous,
            [&callback_स्थितिः, &callback_सन्देशः](wgpu::Queue_Work_समाप्तम्_स्थितिः स्थितिः, wgpu::StringView सन्देशः) {
                callback_स्थितिः  = स्थितिः;
                callback_सन्देशः = std::string(सन्देशः);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_स्थितिः(wait_स्थितिः, callback_स्थितिः, wgpu::Queue_Work_समाप्तम्_स्थितिः::Success,
                                          "Queue wait", "Queue work", callback_सन्देशः.c_str());
}

स्थिर शून्य ggml_backend_webgpu_map_बफर(webgpu_global_context & ctx,
                                           wgpu::बफर_2 &          बफर,
                                           wgpu::Map_बहुलकम्           mode,
                                           size_t                  विस्थापनम्,
                                           size_t                  size) {
    wgpu::Map_Async_स्थितिः callback_स्थितिः = wgpu::Map_Async_स्थितिः::Error;
    std::string          callback_सन्देशः;

    ध्रुव wgpu::Wait_स्थितिः wait_स्थितिः = ctx->instance.WaitAny(
        बफर.MapAsync(mode, विस्थापनम्, size, wgpu::Callback_बहुलकम्::AllowSpontaneous,
                        [&callback_स्थितिः, &callback_सन्देशः](wgpu::Map_Async_स्थितिः स्थितिः, wgpu::StringView सन्देशः) {
                            callback_स्थितिः  = स्थितिः;
                            callback_सन्देशः = std::string(सन्देशः);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_स्थितिः(wait_स्थितिः, callback_स्थितिः, wgpu::Map_Async_स्थितिः::Success,
                                          "Buffer map wait", "Buffer map", callback_सन्देशः.c_str());
}

स्थिर शून्य ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ध्रुव wgpu::Command_बफर commands,
                                                uint32_t &                अङ्कः_inflight_batches) {
    यदि (अङ्कः_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        अङ्कः_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    अङ्कः_inflight_batches++;
}

#यदि_परिभाषितम् GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
स्थिर शून्य ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_बफर_गन्तव्यम्_बफर(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_आकार());
    wgpu::Command_बफर commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_बफर(ctx, ctx->debug_host_buf, wgpu::Map_बहुलकम्::Read, 0, ctx->debug_host_buf.Get_आकार());
    ध्रुव भिन्न * debug_दत्त = (ध्रुव भिन्न *) ctx->debug_host_buf.GetConstMappedRange();
    std::निर्गमः << "debug[0]: " << debug_दत्त[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#यदि_अन्तः

स्थिर webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ध्रुव std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            परिणाम = {};
    std::vector<wgpu::Bind_गणः> bind_groups;
    std::vector<size_t>          param_offsets;
    परिणाम.अङ्कः_kernels = dispatches.size();

    प्रति (size_t i = 0; i < dispatches.size(); i++) {
        ध्रुव webgpu_dispatch_desc & dispatch     = dispatches[i];
        ध्रुव size_t                 param_आकार   = dispatch.params.size() * परिमाण(uint32_t);
        ध्रुव size_t                 param_विस्थापनम् = ctx->param_arena.alloc_slot(param_आकार);

        std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries            = dispatch.bind_गणः_entries;
        uint32_t                          params_binding_अङ्कः = entries.size();
        entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(params_binding_अङ्कः, ctx->param_arena.बफर, param_विस्थापनम्,
                                                            ctx->param_arena.slot_आकार));

        wgpu::Bind_गणः_Descriptor bind_गणः_desc;
        bind_गणः_desc.layout     = dispatch.pipeline.pipeline.Get_Bind_गणः_Layout(0);
        bind_गणः_desc.प्रविष्टिः_गणनम् = entries.size();
        bind_गणः_desc.entries    = entries.दत्त();
        bind_गणः_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.Create_Bind_गणः(&bind_गणः_desc));
        param_offsets.push_back(param_विस्थापनम्);
    }

    प्रति (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_बफर(ctx->param_arena.बफर, param_offsets[i], dispatches[i].params.दत्त(),
                                           dispatches[i].params.size() * परिमाण(uint32_t));
    }

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    प्रति (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_गणनम् + 2 <= WEBGPU_MAX_PROFILE_QUERY_गणनम्);
        ध्रुव uint32_t query_begin = ctx->profile_timestamp_query_गणनम्++;
        ध्रुव uint32_t query_अन्तः   = ctx->profile_timestamp_query_गणनम्++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.query_समुच्चयः                    = ctx->profile_timestamp_query_समुच्चयः;
        ts_writes.beginning_सम्बन्धि_Pass_Write_सूचकांक   = query_begin;
        ts_writes.अन्तः_सम्बन्धि_Pass_Write_सूचकांक         = query_अन्तः;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->सक्रिय_command_encoder.BeginComputePass(&pass_desc);

        pass.समुच्चयः_Pipeline(dispatches[i].pipeline.pipeline);
        pass.समुच्चयः_Bind_गणः(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
        pass.अन्तः_3();
        परिणाम.pipeline_नामानि.push_back(dispatches[i].pipeline.name);
    }
#अन्यथा
    प्रति (size_t i = 0; i < dispatches.size(); i++) {
        यदि (ctx->batch_compute_passes) {
            ctx->सक्रिय_compute_pass.समुच्चयः_Pipeline(dispatches[i].pipeline.pipeline);
            ctx->सक्रिय_compute_pass.समुच्चयः_Bind_गणः(0, bind_groups[i]);
            ctx->सक्रिय_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second,
                                                        1);
        } अन्यथा {
            wgpu::ComputePassEncoder pass = ctx->सक्रिय_command_encoder.BeginComputePass();
            pass.समुच्चयः_Pipeline(dispatches[i].pipeline.pipeline);
            pass.समुच्चयः_Bind_गणः(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
            pass.अन्तः_3();
        }
    }
#यदि_अन्तः

    प्रत्यागम परिणाम;
}

स्थिर webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::Bind_गणः_प्रविष्टिः> bind_गणः_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    प्रत्यागम ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_गणः_entries), { wg_x, wg_y } },
    });
}

स्थिर शून्य ggml_backend_webgpu_बफर_memset(webgpu_global_context & ctx,
                                              wgpu::बफर_2 &          buf,
                                              uint32_t                value,
                                              size_t                  विस्थापनम्,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) विस्थापनम्, (uint32_t) size, value };
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, buf, 0, buf.Get_आकार()) };
    size_t                            अष्टकानि_प्रत्येकम्_wg =
        ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup * ctx->capabilities.memset_अष्टकानि_प्रत्येकम्_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, अष्टकानि_प्रत्येकम्_wg);

    ctx->queue.Write_बफर(ctx->memset_params_buf, 0, params.दत्त(), params.size() * परिमाण(uint32_t));

    wgpu::Bind_गणः_प्रविष्टिः params_प्रविष्टिः = {};
    params_प्रविष्टिः.binding              = 1;
    params_प्रविष्टिः.बफर               = ctx->memset_params_buf;
    params_प्रविष्टिः.विस्थापनम्               = 0;
    params_प्रविष्टिः.size                 = WEBGPU_PARAMS_BUF_आकार_अष्टकानि;
    entries.push_back(params_प्रविष्टिः);

    wgpu::Bind_गणः_Descriptor bind_गणः_desc;
    bind_गणः_desc.layout     = ctx->memset_pipeline.pipeline.Get_Bind_गणः_Layout(0);
    bind_गणः_desc.प्रविष्टिः_गणनम् = entries.size();
    bind_गणः_desc.entries    = entries.दत्त();
    bind_गणः_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::Bind_गणः bind_गणः = ctx->device.Create_Bind_गणः(&bind_गणः_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.समुच्चयः_Pipeline(ctx->memset_pipeline.pipeline);
    pass.समुच्चयः_Bind_गणः(0, bind_गणः);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.अन्तः_3();

    wgpu::Command_बफर              command  = encoder.Finish();
    std::vector<wgpu::Command_बफर> commands = { command };
    ctx->queue.Submit(commands.size(), commands.दत्त());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    प्रत्यागम ctx->name.c_str();
}

स्थिर शून्य ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#यदि_परिभाषितम् GGML_WEBGPU_CPU_PROFILE
    std::निर्गमः << "\n[ggml_webgpu cpu profiling summary]\n";
    द्विगुण योग_cpu = 0.0;
    प्रति (ध्रुव स्वचालित & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        योग_cpu += kv.second;
    }
    std::निर्गमः << "ggml_webgpu: total cpu time: " << योग_cpu << " ms\n";
    std::निर्गमः << "ggml_webgpu: cpu breakdown:\n";
    प्रति (ध्रुव स्वचालित & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        द्विगुण pct = (योग_cpu > 0.0) ? (kv.second / योग_cpu * 100.0) : 0.0;
        std::निर्गमः << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    यदि (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::निर्गमः << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    प्रति (ध्रुव स्वचालित & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        द्विगुण pct = (योग_cpu > 0.0) ? (kv.second / योग_cpu * 100.0) : 0.0;
        std::निर्गमः << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#यदि_अन्तः

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    std::निर्गमः << "\n[ggml_webgpu gpu profiling summary]\n";
    द्विगुण योग_gpu = 0.0;
    प्रति (ध्रुव स्वचालित & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        योग_gpu += kv.second;
    }
    std::निर्गमः << "ggml_webgpu: total gpu time (all shaders): " << योग_gpu << " ms\n";
    std::निर्गमः << "\nggml_webgpu: gpu breakdown:\n";
    प्रति (ध्रुव स्वचालित & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        द्विगुण pct = (योग_gpu > 0.0) ? (kv.second / योग_gpu * 100.0) : 0.0;
        std::निर्गमः << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#यदि_अन्तः

#यदि defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::निर्गमः << "ggml_webgpu: gpu/cpu ratio: " << (योग_cpu > 0.0 ? योग_gpu / योग_cpu : 0.0) << "\n";
#यदि_अन्तः

    विलोपय ctx;
    विलोपय backend;
}

स्थिर webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_आकार(src->type)), (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)), (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_आकार(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_समुच्चयः(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_समुच्चयः_pipeline(shader_lib_ctx);

    स्वचालित *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ध्रुव तार्किक inplace   = decisions->inplace;

    ध्रुव uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    ध्रुव uint32_t dst_type_आकार = (uint32_t) ggml_type_आकार(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[3] / dst_type_आकार),

        (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),

        1u,
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[0] / dst_type_आकार),
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[1] / dst_type_आकार),
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[2] / dst_type_आकार),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries;
    uint32_t                          binding_सूचकांक = 0;
    यदि (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0));
        binding_सूचकांक++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_सूचकांक, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_सूचकांक + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ध्रुव uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),
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

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
    shader_lib_ctx.wg_mem_सीमा_अष्टकानि = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_आकार;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst),
    };

    ध्रुव uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_आकार);
    ध्रुव uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    ध्रुव int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ध्रुव int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ध्रुव int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ध्रुव int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ध्रुव int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ध्रुव int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),

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

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
स्थिर webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    ध्रुव int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ध्रुव int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ध्रुव int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ध्रुव int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ध्रुव int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ध्रुव int32_t d1 = ggml_get_op_params_i32(dst, 5);

    // Scalar params matching conv2d_dw.wgsl (weight src0 [KW,KH,1,C], input src1, output dst).
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),

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

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ध्रुव तार्किक      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    स्वचालित *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    ध्रुव int32_t s0    = ggml_get_op_params_i32(dst, 0);
    ध्रुव int32_t s1    = ggml_get_op_params_i32(dst, 1);
    ध्रुव int32_t p0    = ggml_get_op_params_i32(dst, 2);
    ध्रुव int32_t p1    = ggml_get_op_params_i32(dst, 3);
    ध्रुव int32_t d0    = ggml_get_op_params_i32(dst, 4);
    ध्रुव int32_t d1    = ggml_get_op_params_i32(dst, 5);
    ध्रुव तार्किक    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    ध्रुव uint32_t KW = src0->ne[0];
    ध्रुव uint32_t KH = is_2D ? src0->ne[1] : 1;
    ध्रुव uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    ध्रुव uint32_t IW = src1->ne[0];
    ध्रुव uint32_t IH = is_2D ? src1->ne[1] : 1;
    ध्रुव uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    ध्रुव uint32_t OW = dst->ne[1];
    ध्रुव uint32_t OH = is_2D ? dst->ne[2] : 1;

    ध्रुव uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type));
    ध्रुव uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)) : 0;
    ध्रुव uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type));
    ध्रुव uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type));

    ध्रुव uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_आकार(dst->type));
    ध्रुव uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type));
    ध्रुव uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)) : 0;
    ध्रुव uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),

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

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    स्वचालित *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ध्रुव uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_प्रत्येकम्_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst),
    };

    ध्रुव uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->खण्डम्_आकार_3);
    ध्रुव uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_ssm_अवलोकनम्(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    यदि (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    यदि (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    यदि (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_अवलोकनम्_pipeline(shader_lib_ctx, xbc_overlap);
    स्वचालित *          decisions = static_cast<ggml_webgpu_ssm_अवलोकनम्_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t विस्थापनम्_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    uint32_t विस्थापनम्_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type));
    uint32_t विस्थापनम्_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_आकार(src3->type));
    uint32_t विस्थापनम्_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_आकार(src4->type));
    uint32_t विस्थापनम्_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_आकार(src5->type));
    uint32_t विस्थापनम्_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_आकार(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    यदि (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        विस्थापनम्_x         = ggml_webgpu_tensor_merged_element_विस्थापनम्(src1, xbc_merged_range);
        विस्थापनम्_B         = ggml_webgpu_tensor_merged_element_विस्थापनम्(src4, xbc_merged_range);
    } अन्यथा यदि (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        विस्थापनम्_B         = ggml_webgpu_tensor_merged_element_विस्थापनम्(src4, xbc_merged_range);
        विस्थापनम्_C         = ggml_webgpu_tensor_merged_element_विस्थापनम्(src5, xbc_merged_range);
    } अन्यथा यदि (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        विस्थापनम्_x         = ggml_webgpu_tensor_merged_element_विस्थापनम्(src1, xbc_merged_range);
        विस्थापनम्_B         = ggml_webgpu_tensor_merged_element_विस्थापनम्(src4, xbc_merged_range);
        विस्थापनम्_C         = ggml_webgpu_tensor_merged_element_विस्थापनम्(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_विस्थापनम् = 0;
    size_t xbc_bind_आकार   = 0;
    यदि (xbc_overlap > 0) {
        xbc_bind_विस्थापनम् = xbc_merged_range.विस्थापनम्;
        xbc_bind_आकार   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        विस्थापनम्_x,
        विस्थापनम्_dt,
        विस्थापनम्_A,
        विस्थापनम्_B,
        विस्थापनम्_C,
        विस्थापनम्_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_आकार(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_आकार(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_आकार(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_आकार(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_आकार(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_आकार(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_आकार(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_आकार(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_आकार(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_अङ्कः = 0;

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, src0),
    };
    // xbc_merged binding
    यदि (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(binding_अङ्कः++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_विस्थापनम्, xbc_bind_आकार));
    }
    // x
    यदि (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, src1));
    }
    // B
    यदि (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, src4));
    }
    // C
    यदि (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः++, dst));

    ध्रुव uint32_t योग_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ध्रुव uint32_t max_wg_प्रत्येकम्_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(योग_wg, max_wg_प्रत्येकम्_dim, wg_x, wg_y);

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ध्रुव uint32_t s_v      = (uint32_t) src2->ne[0];
    ध्रुव uint32_t h        = (uint32_t) src2->ne[1];
    ध्रुव uint32_t n_tokens = (uint32_t) src2->ne[2];
    ध्रुव uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ध्रुव uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ध्रुव भिन्न    scale    = 1.0f / sqrtf((भिन्न) s_v);
    uint32_t       scale_u32;
    स्मृतिं_प्रतिलिख(&scale_u32, &scale, परिमाण(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_tokens,
        n_seqs,
        s_v * h * n_tokens * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_आकार(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_आकार(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_आकार(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_आकार(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_आकार(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_आकार(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_आकार(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_आकार(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0), ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, src2), ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 4, src4), ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 6, dst),
    };

    यदि (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 7, dst_fuse));
    }

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

स्थिर std::optional<webgpu_encoded_op> ggml_webgpu_समुच्चयः_पङ्क्तिकाः(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    यदि (ggml_is_रिक्तानि(src) || ggml_is_रिक्तानि(idx)) {
        प्रत्यागम std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_समुच्चयः_पङ्क्तिकाः_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_समुच्चयः_पङ्क्तिकाः_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_आकार(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)), (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)), (uint32_t) (idx->nb[0] / ggml_type_आकार(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_आकार(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_आकार(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst));

    यदि (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(3, ctx->समुच्चयः_पङ्क्तिकाः_dev_error_buf, 0,
                                                            ctx->समुच्चयः_पङ्क्तिकाः_dev_error_buf.Get_आकार()));
    }

    uint32_t तन्तवः;
    यदि (ggml_is_quantized(dst->type)) {
        ध्रुव uint32_t खण्डानि_प्रत्येकम्_row = src->ne[0] / ggml_blck_आकार(dst->type);
        तन्तवः =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_खण्डानि ? (खण्डानि_प्रत्येकम्_row / 2) : खण्डानि_प्रत्येकम्_row);
    } अन्यथा यदि (decisions->vec4) {
        तन्तवः = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } अन्यथा {
        तन्तवः = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(तन्तवः, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
स्थिर std::vector<wgpu::Constant_प्रविष्टिः> ggml_webgpu_wg_आकार_प्रविष्टिः(uint32_t wg_आकार) {
    std::vector<wgpu::Constant_प्रविष्टिः> constants(1);
    constants[0].कुञ्चिका   = "wg_size";
    constants[0].value = wg_आकार;
    प्रत्यागम constants;
}

स्थिर webgpu_encoded_op ggml_webgpu_get_पङ्क्तिकाः(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ध्रुव तार्किक float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = शून्यसूचकः;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    ध्रुव uint32_t विस्थापनम्_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type));
    ध्रुव uint32_t विस्थापनम्_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type));
    ध्रुव uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_आकार(src->type));
    ध्रुव uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_आकार(src->type));
    ध्रुव uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_आकार(src->type));

    ध्रुव तार्किक vec4_aligned = विस्थापनम्_src % 4 == 0 && विस्थापनम्_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_पङ्क्तिकाः_pipeline(shader_lib_ctx, vec4_aligned);
    स्वचालित *          decisions = static_cast<ggml_webgpu_get_पङ्क्तिकाः_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { विस्थापनम्_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_आकार(idx->type)),
                                     विस्थापनम्_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_आकार(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_आकार(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_आकार(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst) };

    uint32_t खण्डानि_प्रत्येकम्_row = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t योग_पङ्क्तिकाः     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t योग_तन्तवः  = float_parallel ? खण्डानि_प्रत्येकम्_row * योग_पङ्क्तिकाः : योग_पङ्क्तिकाः;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(योग_तन्तवः, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर शून्य ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ध्रुव size_t dst_विस्थापनम्           = ggml_webgpu_tensor_विस्थापनम्(dst);
    ध्रुव size_t q8_src1_align_विस्थापनम् = ROUNDUP_POW2(
        dst_विस्थापनम् + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment);
    ध्रुव size_t q8_src1_binding_आकार = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> q8_entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src1),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_विस्थापनम्, q8_src1_binding_आकार)
    };

    स्वचालित q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_आकार     = q8_decisions->wg_आकार;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ध्रुव uint32_t wg_प्रत्येकम्_vec     = (src0->ne[0] / 4 + (q8_wg_आकार - 1)) / q8_wg_आकार;
    ध्रुव uint32_t q8_योग_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_प्रत्येकम्_vec;
    ध्रुव uint32_t max_wg_प्रत्येकम्_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension;
    compute_2d_workgroups(q8_योग_wg, max_wg_प्रत्येकम्_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

स्थिर webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    तार्किक use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    तार्किक use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_गुणनफलम्,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_आकार              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_आकार        = ctx->global_ctx->capabilities.min_subgroup_आकार;
    shader_lib_ctx.max_subgroup_आकार        = ctx->global_ctx->capabilities.max_subgroup_आकार;
    shader_lib_ctx.supports_dot_गुणनफलम्     = ctx->global_ctx->capabilities.supports_dot_गुणनफलम्;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    ध्रुव तार्किक src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    यदि (use_mat_vec) {
        यदि (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } अन्यथा {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t विस्थापनम्_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t विस्थापनम्_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    size_t   merged_विस्थापनम् = 0;
    size_t   merged_आकार   = 0;
    यदि (src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_विस्थापनम् = merged_range.विस्थापनम्;
        merged_आकार   = merged_range.size;
        विस्थापनम्_src0   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src0, merged_range);
        विस्थापनम्_src1   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { विस्थापनम्_src0,
                                     विस्थापनम्_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {};
    यदि (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0));
        स्वचालित & mmvq_qq8_प्रविष्टिः = dispatches[0].bind_गणः_entries[1];
        entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_प्रविष्टिः.विस्थापनम्,
                                                            mmvq_qq8_प्रविष्टिः.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst));
    } अन्यथा यदि (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src0), merged_विस्थापनम्, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ध्रुव uint32_t max_wg_प्रत्येकम्_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension;

    यदि (use_mat_vec) {
        स्वचालित * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_प्रत्येकम्_wg);
        uint32_t योग_wg      = output_groups * batches;
        compute_2d_workgroups(योग_wg, max_wg_प्रत्येकम्_dim, wg_x, wg_y);
    } अन्यथा {
        स्वचालित * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        यदि (decisions->use_subgroup_matrix) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_matrix_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_matrix_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } अन्यथा {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_आकार_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_आकार_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t योग_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(योग_wg, max_wg_प्रत्येकम्_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    प्रत्यागम ggml_backend_webgpu_build_multi(ctx, dispatches);
}

स्थिर webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    ध्रुव uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ध्रुव uint32_t param_n_expert_प्रयुक्तम् = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_प्रयुक्तम्,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_विस्थापनम्(ctx, src0),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src0)),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_विस्थापनम्(ctx, src1),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src1)),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_विस्थापनम्(ctx, src2),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src2)),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_विस्थापनम्(ctx, dst),
                                          ggml_webgpu_tensor_binding_आकार(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    स्वचालित * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ध्रुव uint32_t max_wg_प्रत्येकम्_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_प्रत्येकम्_wg);
    uint32_t       योग_wg       = output_groups * param_n_expert_प्रयुक्तम्;
    compute_2d_workgroups(योग_wg, max_wg_प्रत्येकम्_dim, wg_x, wg_y);

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
स्थिर uint32_t ggml_webgpu_mul_mat_id_max_tiles(ध्रुव ggml_tensor * dst) {
    ध्रुव uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    ध्रुव uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    प्रत्यागम CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_आकार_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
स्थिर size_t ggml_webgpu_mul_mat_id_tiles_nbytes(ध्रुव ggml_tensor * dst) {
    प्रत्यागम (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * परिमाण(uint32_t);
}

स्थिर webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    यदि (dst->ne[2] == 1) {
        प्रत्यागम ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ध्रुव uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ध्रुव uint32_t param_n_expert_प्रयुक्तम् = (uint32_t) dst->ne[1];
    ध्रुव uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)),
        param_n_expert,
        param_n_expert_प्रयुक्तम्,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_आकार(src2->type)),
    };

    ध्रुव size_t dst_विस्थापनम्       = ggml_webgpu_tensor_विस्थापनम्(dst);
    ध्रुव size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * परिमाण(uint32_t);
    ध्रुव size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ध्रुव size_t ids_cache_align_विस्थापनम् = ROUNDUP_POW2(
        dst_विस्थापनम् + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment);
    ध्रुव size_t ids_tiles_align_विस्थापनम् =
        ROUNDUP_POW2(ids_cache_align_विस्थापनम् + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment);

    ध्रुव size_t ids_cache_bind_आकार = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ध्रुव size_t ids_tiles_bind_आकार = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> gather_entries = {
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_विस्थापनम्(ctx, src2),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src2)),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_विस्थापनम्, ids_cache_bind_आकार),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_विस्थापनम्, ids_tiles_bind_आकार),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_प्रयुक्तम्,
        param_n_tokens,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> main_entries = {
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_विस्थापनम्(ctx, src0),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src0)),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_विस्थापनम्(ctx, src1),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src1)),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_विस्थापनम्(ctx, dst),
                                          ggml_webgpu_tensor_binding_आकार(ctx, dst)),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_विस्थापनम्, ids_cache_bind_आकार),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_विस्थापनम्, ids_tiles_bind_आकार),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    स्वचालित * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_आकार_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_आकार_N);

    ध्रुव uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_आकार_m;
    ध्रुव uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ध्रुव uint32_t योग_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    प्रत्यागम ggml_backend_webgpu_build_multi(ctx, dispatches);
}

संरचना ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries;
    size_t                            kv_bind_विस्थापनम् = 0;
    size_t                            kv_bind_आकार   = 0;
    तार्किक                              has_आवरणम्       = असत्य;
    तार्किक                              has_sinks      = असत्य;
    तार्किक                              kv_overlap     = असत्य;
};

स्थिर तार्किक ggml_webgpu_flash_attn_use_vec_मार्गः(ध्रुव webgpu_global_context & global_ctx,
                                                ध्रुव ggml_tensor *           Q,
                                                ध्रुव ggml_tensor *           K,
                                                ध्रुव ggml_tensor *           V) {
    ध्रुव size_t storage_विस्थापनम्_alignment = global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment;

    ध्रुव तार्किक k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_विस्थापनम्_alignment);
    ध्रुव तार्किक v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_विस्थापनम्_alignment);

    ध्रुव uint32_t k_vec_शीर्ष_align =
        ggml_is_quantized(K->type) ? ggml_blck_आकार(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः;
    ध्रुव uint32_t v_vec_शीर्ष_align =
        ggml_is_quantized(V->type) ? ggml_blck_आकार(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः;
    ध्रुव तार्किक kv_vec_शीर्ष_dims_aligned = Q->ne[0] % k_vec_शीर्ष_align == 0 && V->ne[0] % v_vec_शीर्ष_align == 0;

    प्रत्यागम global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_शीर्ष_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

स्थिर ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    आवरणम्,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    भिन्न scale         = ggml_get_op_params_f32(dst, 0);
    भिन्न max_bias      = ggml_get_op_params_f32(dst, 1);
    भिन्न logit_softcap = ggml_get_op_params_f32(dst, 2);
    यदि (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    भिन्न n_शीर्ष_log2 = भिन्न(1u << (uint32_t) floor(log2(Q->ne[2])));
    भिन्न m0          = powf(2.0f, -(max_bias) / n_शीर्ष_log2);
    भिन्न m1          = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = आवरणम्;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    op.shader_lib_ctx.max_wg_आकार              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
    op.shader_lib_ctx.wg_mem_सीमा_अष्टकानि       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_आकार;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_आकार        = ctx->global_ctx->capabilities.min_subgroup_आकार;
    op.shader_lib_ctx.max_subgroup_आकार        = ctx->global_ctx->capabilities.max_subgroup_आकार;

    op.has_आवरणम्   = आवरणम् != शून्यसूचकः;
    op.has_sinks  = sinks != शून्यसूचकः;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t विस्थापनम्_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_आकार(K->type));
    uint32_t विस्थापनम्_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_आकार(V->type));
    यदि (op.kv_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_विस्थापनम्                                   = merged_range.विस्थापनम्;
        op.kv_bind_आकार                                     = merged_range.size;
        विस्थापनम्_k                                            = ggml_webgpu_tensor_merged_element_विस्थापनम्(K, merged_range);
        विस्थापनम्_v                                            = ggml_webgpu_tensor_merged_element_विस्थापनम्(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_आकार(Q->type)),
        विस्थापनम्_k,
        विस्थापनम्_v,
        op.has_आवरणम् ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, आवरणम्) / ggml_type_आकार(आवरणम्->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_आकार(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_आकार(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_आकार(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_आकार(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_आकार(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_आकार(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_आकार(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_आकार(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_आकार(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_आकार(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_आवरणम् ? (uint32_t) (आवरणम्->nb[3] / ggml_type_आकार(आवरणम्->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_शीर्ष_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, Q),
    };
    यदि (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(K), op.kv_bind_विस्थापनम्, op.kv_bind_आकार));
    } अन्यथा {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, V));
    }
    uint32_t binding_सूचकांक = op.kv_overlap ? 2u : 3u;
    यदि (op.has_आवरणम्) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_सूचकांक++, आवरणम्));
    }
    यदि (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_सूचकांक++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_सूचकांक++, dst));

    प्रत्यागम op;
}

स्थिर uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ध्रुव uint64_t kv_विस्तृतिः = (uint64_t) kv_tile;
    यावत् ((2u * nwg * kv_विस्तृतिः) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    प्रत्यागम std::min(nwg, vec_nwg_cap);
}

स्थिर webgpu_encoded_op ggml_webgpu_flash_attn_प्रत्यक्षम्(webgpu_context & ctx, ध्रुव ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    स्वचालित *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_प्रत्येकम्_शीर्ष = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_प्रत्येकम्_शीर्ष * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

स्थिर webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             आवरणम्,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    स्वचालित *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::बफर_2 blk_buf         = {};
    uint64_t     blk_आकार_अष्टकानि  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_गणनम् = 0;

    ध्रुव uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_आकार;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ध्रुव uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ध्रुव तार्किक     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_आधारः  = 0;
    uint64_t     tmp_आकार_अष्टकानि  = 0;
    wgpu::बफर_2 tmp_buf         = {};
    uint64_t     tmp_bind_विस्थापनम् = 0;
    uint64_t     tmp_bind_आकार   = 0;
    ध्रुव size_t align_अष्टकानि     = ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment;
    ध्रुव size_t dst_विस्थापनम्      = ggml_webgpu_tensor_विस्थापनम्(dst);
    size_t       scratch_विस्थापनम्  = ROUNDUP_POW2(dst_विस्थापनम् + ggml_nbytes(dst), align_अष्टकानि);

    यदि (use_vec_reduce) {
        ध्रुव uint64_t tmp_दत्त_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ध्रुव uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_आधारः                 = tmp_दत्त_elems;
        tmp_आकार_अष्टकानि =
            ROUNDUP_POW2((tmp_दत्त_elems + tmp_stats_elems) * परिमाण(भिन्न), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_आधारः <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_विस्थापनम् = scratch_विस्थापनम्;
        tmp_bind_आकार   = tmp_आकार_अष्टकानि;
        scratch_विस्थापनम्  = ROUNDUP_POW2(scratch_विस्थापनम् + tmp_आकार_अष्टकानि, align_अष्टकानि);
    } अन्यथा {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_आकार_अष्टकानि  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_विस्थापनम् = scratch_विस्थापनम्;
        tmp_bind_आकार   = tmp_आकार_अष्टकानि;
        scratch_विस्थापनम्  = ROUNDUP_POW2(scratch_विस्थापनम् + tmp_आकार_अष्टकानि, align_अष्टकानि);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> blk_entries;
    यदि (op.has_आवरणम्) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ध्रुव uint32_t stride_mask3 = (uint32_t) (आवरणम्->nb[3] / ggml_type_आकार(आवरणम्->type));
        blk_batch_गणनम्             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ध्रुव uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_गणनम्;
        blk_आकार_अष्टकानि              = ROUNDUP_POW2(blk_elems * परिमाण(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        ध्रुव ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, आवरणम्) / ggml_type_आकार(आवरणम्->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(आवरणम्),
                                              ggml_webgpu_tensor_align_विस्थापनम्(ctx, आवरणम्),
                                              ggml_webgpu_tensor_binding_आकार(ctx, आवरणम्)),
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, blk_buf, scratch_विस्थापनम्, blk_आकार_अष्टकानि),
        };
        scratch_विस्थापनम् = ROUNDUP_POW2(scratch_विस्थापनम् + blk_आकार_अष्टकानि, align_अष्टकानि);
    }

    std::vector<uint32_t> विभज_params = op.params;
    यदि (op.has_आवरणम्) {
        विभज_params.push_back(0u);                     // blk_base
        विभज_params.push_back(blk_nblk0);              // blk_nblk0
        विभज_params.push_back(blk_nblk1);              // blk_nblk1
    }
    विभज_params.push_back(0u);                         // tmp_data_base
    विभज_params.push_back((uint32_t) tmp_stats_आधारः);  // tmp_stats_base
    विभज_params.push_back(nwg);                        // nwg

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> विभज_entries = {
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_विस्थापनम्(ctx, Q),
                                          ggml_webgpu_tensor_binding_आकार(ctx, Q)),
    };
    यदि (op.kv_overlap) {
        विभज_entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(K), op.kv_bind_विस्थापनम्, op.kv_bind_आकार));
    } अन्यथा {
        विभज_entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_विस्थापनम्(ctx, K),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, K)));
        विभज_entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_विस्थापनम्(ctx, V),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, V)));
    }
    uint32_t विभज_binding_सूचकांक = op.kv_overlap ? 2u : 3u;
    यदि (op.has_आवरणम्) {
        विभज_entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(विभज_binding_सूचकांक++, ggml_webgpu_tensor_buf(आवरणम्),
                                                                  ggml_webgpu_tensor_align_विस्थापनम्(ctx, आवरणम्),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, आवरणम्)));
    }
    यदि (op.has_sinks) {
        विभज_entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(विभज_binding_सूचकांक++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_विस्थापनम्(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, sinks)));
    }
    यदि (op.has_आवरणम्) {
        विभज_entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(विभज_binding_सूचकांक++, blk_buf, blk_entries[1].विस्थापनम्, blk_आकार_अष्टकानि));
    }
    विभज_entries.push_back(
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(विभज_binding_सूचकांक++, tmp_buf, tmp_bind_विस्थापनम्, tmp_bind_आकार));
    विभज_entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(विभज_binding_सूचकांक++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_विस्थापनम्(ctx, dst),
                                                              ggml_webgpu_tensor_binding_आकार(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> reduce_entries;
    यदि (use_vec_reduce) {
        ध्रुव uint32_t reduce_sg_आकार = ctx->global_ctx->capabilities.max_subgroup_आकार;
        ध्रुव uint32_t reduce_wg_आकार = std::max(
            reduce_sg_आकार,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_आकार,
                                          ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_आकार                    = reduce_wg_आकार;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_आधारः,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, tmp_buf, tmp_bind_विस्थापनम्, tmp_आकार_अष्टकानि),
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_विस्थापनम्(ctx, dst),
                                              ggml_webgpu_tensor_binding_आकार(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ध्रुव uint64_t विभज_wg_योग = (uint64_t) wg_x * nwg;
    GGML_ASSERT(विभज_wg_योग <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    यदि (op.has_आवरणम्) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_गणनम् }
        });
    }
    dispatches.push_back({
        pipeline, std::move(विभज_params), std::move(विभज_entries), { (uint32_t) विभज_wg_योग, 1u }
    });
    यदि (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    प्रत्यागम ggml_backend_webgpu_build_multi(ctx, dispatches);
}

स्थिर webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    आवरणम्,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, आवरणम्, sinks, dst);
    यदि (ggml_webgpu_flash_attn_use_vec_मार्गः(ctx->global_ctx, Q, K, V)) {
        प्रत्यागम ggml_webgpu_flash_attn_vec(ctx, Q, K, V, आवरणम्, sinks, dst, std::move(op));
    }
    प्रत्यागम ggml_webgpu_flash_attn_प्रत्यक्षम्(ctx, op);
}

स्थिर webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    तार्किक is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = शून्यसूचकः;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    स्वचालित *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ध्रुव तार्किक inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_आकार(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    यदि (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        विकल्प (unary_op) {
            पक्ष GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    भिन्न alpha_n = ggml_get_op_params_f32(dst, 1);
                    भिन्न alpha_p = ggml_get_op_params_f32(dst, 2);
                    भिन्न beta    = ggml_get_op_params_f32(dst, 3);
                    भिन्न eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    भङ्ग;
                }
            स्वभाव:
                भङ्ग;
        }
    } अन्यथा यदि (dst->op == GGML_OP_CLAMP) {
        भिन्न clamp_min = ggml_get_op_params_f32(dst, 0);
        भिन्न clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } अन्यथा यदि (dst->op == GGML_OP_FILL) {
        भिन्न fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, effective_src),
    };
    यदि (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_द्विमान_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    ध्रुव तार्किक      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_द्विमान_pipeline(shader_lib_ctx, src_overlap);
    स्वचालित *          decisions   = static_cast<ggml_webgpu_द्विमान_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_विस्थापनम् = ggml_webgpu_tensor_align_विस्थापनम्(ctx, src0);
    size_t src1_webgpu_tensor_align_विस्थापनम् = ggml_webgpu_tensor_align_विस्थापनम्(ctx, src1);

    uint32_t विस्थापनम्_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t विस्थापनम्_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    size_t   merged_विस्थापनम् = 0;
    size_t   merged_आकार   = 0;
    यदि (decisions->src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_विस्थापनम् = merged_range.विस्थापनम्;
        merged_आकार   = merged_range.size;
        विस्थापनम्_src0   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src0, merged_range);
        विस्थापनम्_src1   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        विस्थापनम्_src0,
        विस्थापनम्_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries;

    यदि (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src0), merged_विस्थापनम्, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_विस्थापनम्,
                                                            ggml_webgpu_tensor_binding_आकार(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_विस्थापनम्,
                                                            ggml_webgpu_tensor_binding_आकार(ctx, src1)));
        यदि (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_सङ्कलय_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_सङ्कलय_id_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_आकार(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_आकार(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, src2));

    यदि (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       योग_wg       = ggml_nrows(dst);
    ध्रुव uint32_t max_wg_प्रत्येकम्_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension;
    compute_2d_workgroups(योग_wg, max_wg_प्रत्येकम्_dim, wg_x, wg_y);

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    यदि (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        प्रत्यागम {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    ध्रुव तार्किक      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    स्वचालित *          decisions   = static_cast<ggml_webgpu_द्विमान_shader_decisions *>(pipeline.context.get());

    uint32_t विस्थापनम्_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t विस्थापनम्_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    size_t   merged_विस्थापनम् = 0;
    size_t   merged_आकार   = 0;
    यदि (decisions->src_overlap) {
        यदि (ggml_nbytes(src0) == 0) {
            merged_विस्थापनम् = ggml_webgpu_tensor_align_विस्थापनम्(ctx, src1);
            merged_आकार   = ggml_webgpu_tensor_binding_आकार(ctx, src1);
            विस्थापनम्_src0   = 0;
            विस्थापनम्_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
        } अन्यथा यदि (ggml_nbytes(src1) == 0) {
            merged_विस्थापनम् = ggml_webgpu_tensor_align_विस्थापनम्(ctx, src0);
            merged_आकार   = ggml_webgpu_tensor_binding_आकार(ctx, src0);
            विस्थापनम्_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
            विस्थापनम्_src1   = 0;
        } अन्यथा {
            ध्रुव ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_विस्थापनम् = merged_range.विस्थापनम्;
            merged_आकार   = merged_range.size;
            विस्थापनम्_src0   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src0, merged_range);
            विस्थापनम्_src1   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     विस्थापनम्_src0,
                                     विस्थापनम्_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {};
    यदि (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src0), merged_विस्थापनम्, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_आकार(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    स्वचालित *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    यदि (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } अन्यथा यदि (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } अन्यथा {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t विस्थापनम्_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_आकार(rn_src->type));
    uint32_t विस्थापनम्_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_आकार(mul_src->type));
    size_t merged_विस्थापनम् = 0;
    size_t merged_आकार   = 0;

    std::vector<uint32_t> params = {
        विस्थापनम्_rn_src,
        विस्थापनम्_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_आकार(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_आकार(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_आकार(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_आकार(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_आकार(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_आकार(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
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

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    ध्रुव तार्किक      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    स्वचालित *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    यदि (decisions->src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_विस्थापनम्  = merged_range.विस्थापनम्;
        merged_आकार    = merged_range.size;
        विस्थापनम्_rn_src  = ggml_webgpu_tensor_merged_element_विस्थापनम्(rn_src, merged_range);
        विस्थापनम्_mul_src = ggml_webgpu_tensor_merged_element_विस्थापनम्(mul_src, merged_range);
        params[0]      = विस्थापनम्_rn_src;
        params[1]      = विस्थापनम्_mul_src;
    }

    यदि (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, mul_src));
    } अन्यथा यदि (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(rn_src), merged_विस्थापनम्, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x,
                          wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
    स्वचालित *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src) };
    यदि (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x,
                          wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    स्वचालित * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ध्रुव तार्किक inplace         = decisions->inplace;
    ध्रुव पूर्णाङ्क  has_freq_गुणकः = (src2 != शून्यसूचकः);

    ध्रुव पूर्णाङ्क n_dims     = ((int32_t *) dst->op_params)[1];
    ध्रुव पूर्णाङ्क mode       = ((int32_t *) dst->op_params)[2];
    ध्रुव पूर्णाङ्क n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ध्रुव पूर्णाङ्क n_offs     = ((int32_t *) dst->op_params)[15];

    भिन्न freq_आधारः;
    भिन्न freq_scale;
    भिन्न ext_गुणकः;
    भिन्न attn_गुणकः;
    भिन्न beta_fast;
    भिन्न beta_slow;
    स्मृतिं_प्रतिलिख(&freq_आधारः, (int32_t *) dst->op_params + 5, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&freq_scale, (int32_t *) dst->op_params + 6, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&ext_गुणकः, (int32_t *) dst->op_params + 7, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&attn_गुणकः, (int32_t *) dst->op_params + 8, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_fast, (int32_t *) dst->op_params + 9, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_slow, (int32_t *) dst->op_params + 10, परिमाण(भिन्न));

    पूर्णाङ्क sections[4];
    स्मृतिं_प्रतिलिख(sections, (int32_t *) dst->op_params + 11, 4 * परिमाण(पूर्णाङ्क));

    भिन्न theta_scale = powf(freq_आधारः, -2.0f / n_dims);

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_आधारः, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        src2 != शून्यसूचकः ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_गुणकः),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_गुणकः),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries     = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    यदि (has_freq_गुणकः) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 2, src2));
    }
    यदि (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    ध्रुव तार्किक      src_overlap = src1 != शून्यसूचकः && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    स्वचालित * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ध्रुव पूर्णाङ्क विभज = (src1 != शून्यसूचकः);

    uint32_t विस्थापनम्_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t विस्थापनम्_src1 =
        src1 != शून्यसूचकः ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)) : 0;
    size_t merged_विस्थापनम् = 0;
    size_t merged_आकार   = 0;
    यदि (decisions->src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_विस्थापनम् = merged_range.विस्थापनम्;
        merged_आकार   = merged_range.size;
        विस्थापनम्_src0   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src0, merged_range);
        विस्थापनम्_src1   = ggml_webgpu_tensor_merged_element_विस्थापनम्(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        विस्थापनम्_src0,
        विस्थापनम्_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
        src1 != शून्यसूचकः ? (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        src1 != शून्यसूचकः ? (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        src1 != शून्यसूचकः ? (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  //सीमा
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries;
    uint32_t                          dst_binding = 1;
    यदि (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(0, ggml_webgpu_tensor_buf(src0), merged_विस्थापनम्, merged_आकार));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src0));
    }
    यदि (विभज && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = शून्यसूचकः;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    स्वचालित *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src) };

    यदि (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    स्वचालित *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ध्रुव तार्किक inplace     = decisions->inplace;
    ध्रुव पूर्णाङ्क  has_आवरणम्    = (src1 != शून्यसूचकः);
    ध्रुव पूर्णाङ्क  has_sink    = (src2 != शून्यसूचकः);
    भिन्न      max_bias    = ggml_get_op_params_f32(dst, 1);
    भिन्न      n_शीर्ष_log2 = भिन्न(1u << (uint32_t) floor(log2(src0->ne[2])));
    भिन्न      m0          = powf(2.0f, -(max_bias) / n_शीर्ष_log2);
    भिन्न      m1          = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        has_आवरणम् ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
        has_आवरणम् ? (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)) : 0,
        has_आवरणम् ? (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)) : 0,
        has_आवरणम् ? (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_आवरणम् ? (uint32_t) src1->ne[2] : 0,
        has_आवरणम् ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_शीर्ष_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries     = { ggml_webgpu_make_bind_गणः_प्रविष्टिः(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_विस्थापनम्(ctx, src0),
        ggml_webgpu_tensor_binding_आकार(ctx, src0)) };
    uint32_t                          binding_अङ्कः = 1;
    यदि (has_आवरणम्) {
        entries.push_back(ggml_webgpu_make_bind_गणः_प्रविष्टिः(binding_अङ्कः, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_विस्थापनम्(ctx, src1),
                                                            ggml_webgpu_tensor_binding_आकार(ctx, src1)));
        binding_अङ्कः++;
    }
    यदि (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः, src2));
        binding_अङ्कः++;
    }
    यदि (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, binding_अङ्कः, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = ggml_nrows(dst);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

स्थिर webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    तार्किक is_शिखरम्_k = dst->op == GGML_OP_शिखरम्_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = शून्यसूचकः;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
    shader_lib_ctx.wg_mem_सीमा_अष्टकानि = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_आकार;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    स्वचालित * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    ध्रुव uint32_t src_ne0 = (uint32_t) src->ne[0];
    ध्रुव uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ध्रुव uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_आकार);
    ध्रुव uint32_t खण्डम्_आकार_3 =
        is_शिखरम्_k ? std::min(argsort_decisions->wg_आकार, (uint32_t) dst->ne[0]) : argsort_decisions->wg_आकार;
    uint32_t बहिर्गमनम्_ne0 = src_ne0;
    यदि (is_शिखरम्_k) {
        यदि (npr > 1) {
            ध्रुव uint32_t चरमः_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_आकार;
            बहिर्गमनम्_ne0                  = (npr - 1) * खण्डम्_आकार_3 + std::min(चरमः_tile, खण्डम्_आकार_3);
        } अन्यथा {
            बहिर्गमनम्_ne0 = खण्डम्_आकार_3;
        }
    }

    uint32_t merge_len    = खण्डम्_आकार_3;
    uint32_t merge_passes = 0;
    यावत् (merge_len < बहिर्गमनम्_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    ध्रुव तार्किक start_अन्तर्_tmp = (merge_passes % 2) == 1;

    ध्रुव size_t dst_विस्थापनम् = ggml_webgpu_tensor_विस्थापनम्(dst);
    ध्रुव size_t idx_nbytes = बहिर्गमनम्_ne0 * ggml_nrows(dst) * परिमाण(int32_t);
    ध्रुव size_t tmp_विस्थापनम् =
        ROUNDUP_POW2(dst_विस्थापनम् + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment);
    ध्रुव size_t tmp_binding_आकार = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ध्रुव size_t dst_binding_आकार =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ध्रुव uint32_t विस्थापनम्_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type));
    ध्रुव uint32_t विस्थापनम्_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type));
    ध्रुव uint32_t विस्थापनम्_tmp  = 0;
    ध्रुव uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_आकार(src->type));
    ध्रुव uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_आकार(src->type));
    ध्रुव uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_आकार(src->type));
    ध्रुव uint32_t stride_idx1 = बहिर्गमनम्_ne0;
    ध्रुव uint32_t stride_idx2 = बहिर्गमनम्_ne0 * (uint32_t) dst->ne[1];
    ध्रुव uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ध्रुव uint32_t init_विस्थापनम्       = start_अन्तर्_tmp ? विस्थापनम्_tmp : विस्थापनम्_dst;
    ध्रुव size_t   init_align_विस्थापनम् = start_अन्तर्_tmp ? tmp_विस्थापनम् : ggml_webgpu_tensor_align_विस्थापनम्(ctx, dst);
    ध्रुव size_t   init_binding_आकार = start_अन्तर्_tmp ? tmp_binding_आकार : dst_binding_आकार;

    std::vector<uint32_t> init_params = {
        विस्थापनम्_src,  init_विस्थापनम्, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], बहिर्गमनम्_ne0,
        खण्डम्_आकार_3,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ध्रुव uint32_t योग_wg_init  = npr * nrows;
    ध्रुव uint32_t max_wg_प्रत्येकम्_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension;
    compute_2d_workgroups(योग_wg_init, max_wg_प्रत्येकम्_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> init_entries = {
        ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
        ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(dst), init_align_विस्थापनम्, init_binding_आकार)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    यदि (merge_passes == 0) {
        प्रत्यागम ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    तार्किक     अन्तर्_is_tmp = start_अन्तर्_tmp;
    uint32_t len       = खण्डम्_आकार_3;
    यावत् (len < बहिर्गमनम्_ne0) {
        ध्रुव uint32_t nm = CEIL_DIV(बहिर्गमनम्_ne0, 2 * len);

        ध्रुव तार्किक     बहिर्गमनम्_is_tmp  = !अन्तर्_is_tmp;
        ध्रुव uint32_t विस्थापनम्_अन्तर्   = अन्तर्_is_tmp ? विस्थापनम्_tmp : विस्थापनम्_dst;
        ध्रुव uint32_t विस्थापनम्_बहिर्गमनम्  = बहिर्गमनम्_is_tmp ? विस्थापनम्_tmp : विस्थापनम्_dst;
        ध्रुव size_t   align_अन्तर्    = अन्तर्_is_tmp ? tmp_विस्थापनम् : ggml_webgpu_tensor_align_विस्थापनम्(ctx, dst);
        ध्रुव size_t   align_बहिर्गमनम्   = बहिर्गमनम्_is_tmp ? tmp_विस्थापनम् : ggml_webgpu_tensor_align_विस्थापनम्(ctx, dst);
        ध्रुव size_t   आकार_अन्तर्     = अन्तर्_is_tmp ? tmp_binding_आकार : dst_binding_आकार;
        ध्रुव size_t   आकार_बहिर्गमनम्    = बहिर्गमनम्_is_tmp ? tmp_binding_आकार : dst_binding_आकार;
        ध्रुव uint32_t शिखरम्_k_बहिर्गमनम्   = (is_शिखरम्_k && nm == 1) ? (uint32_t) dst->ne[0] : बहिर्गमनम्_ne0;
        ध्रुव uint32_t stride_out1 = शिखरम्_k_बहिर्गमनम्;
        ध्रुव uint32_t stride_out2 = शिखरम्_k_बहिर्गमनम् * (uint32_t) dst->ne[1];
        ध्रुव uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> merge_params = { विस्थापनम्_src,
                                               विस्थापनम्_अन्तर्,
                                               विस्थापनम्_बहिर्गमनम्,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               बहिर्गमनम्_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               शिखरम्_k_बहिर्गमनम्,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::Bind_गणः_प्रविष्टिः> merge_entries = {
            ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(1, ggml_webgpu_tensor_buf(dst), align_अन्तर्, आकार_अन्तर्),
            ggml_webgpu_make_bind_गणः_प्रविष्टिः(2, ggml_webgpu_tensor_buf(dst), align_बहिर्गमनम्, आकार_बहिर्गमनम्)
        };

        uint32_t       wg_x_merge;
        uint32_t       wg_y_merge;
        ध्रुव uint32_t योग_wg_merge = nm * nrows;
        compute_2d_workgroups(योग_wg_merge, max_wg_प्रत्येकम्_dim, wg_x_merge, wg_y_merge);

        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        अन्तर्_is_tmp = !अन्तर्_is_tmp;
    }

    प्रत्यागम ggml_backend_webgpu_build_multi(ctx, dispatches);
}

स्थिर webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = शून्यसूचकः;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        योग_wg = ggml_nrows(dst);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_योग_पङ्क्तिकाः(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    तार्किक                  योग_योग = dst->op == GGML_OP_योग;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     योग_योग ? 0 : (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
                                     योग_योग ? 0 : (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
                                     योग_योग ? 0 : (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),
                                     योग_योग ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     योग_योग ? 1 : (uint32_t) src->ne[1],
                                     योग_योग ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_योग_पङ्क्तिकाः_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = योग_योग ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);
    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर तार्किक ggml_webgpu_can_fuse_rms_norm_mul(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    यदि (!ggml_can_fuse(cgraph, ग्रन्थिः_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        प्रत्यागम असत्य;
    }

    // additional constraints specific to this fusion
    ध्रुव ggml_tensor * rms_norm = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * mul      = cgraph->nodes[ग्रन्थिः_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    यदि (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    यदि (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        प्रत्यागम असत्य;
    }
    // rms_norm shader assumes contiguous rows
    यदि (!ggml_is_contiguous_पङ्क्तिकाः(mul->src[0]) || !ggml_is_contiguous_पङ्क्तिकाः(mul->src[1])) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_webgpu_can_fuse_gdn_cache(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, पूर्णाङ्क & अङ्कः_encoded_ops) {
    ध्रुव ggml_tensor * gdn = cgraph->nodes[ग्रन्थिः_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    यदि (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        प्रत्यागम असत्य;
    }

    ध्रुव ggml_tensor * src_v     = gdn->src[2];
    ध्रुव int64_t       S_v       = src_v->ne[0];
    ध्रुव int64_t       H         = src_v->ne[1];
    ध्रुव int64_t       n_tokens  = src_v->ne[2];
    ध्रुव int64_t       n_seqs    = src_v->ne[3];
    ध्रुव int64_t       D         = S_v * S_v * H;
    ध्रुव int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    ध्रुव int64_t       n_written = std::min<int64_t>(n_tokens, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ध्रुव size_t पुच्छ_off = ggml_row_आकार(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ध्रुव ggml_tensor * cpy     = शून्यसूचकः;
    पूर्णाङ्क                 cpy_idx = 0;
    प्रति (पूर्णाङ्क j = ग्रन्थिः_idx + 1; j < cgraph->n_nodes && cpy == शून्यसूचकः; ++j) {
        ध्रुव ggml_tensor * n = cgraph->nodes[j];
        यदि (ggml_op_is_रिक्तानि(n->op) || ggml_is_रिक्तानि(n)) {
            अनुवर्त;
        }
        यदि (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            प्रत्यागम असत्य;
        }
        cpy     = n;
        cpy_idx = j;
    }
    यदि (cpy == शून्यसूचकः) {
        प्रत्यागम असत्य;
    }

    ध्रुव ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ध्रुव ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    यदि (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != पुच्छ_off ||
        !ggml_is_contiguous(cpy_src)) {
        प्रत्यागम असत्य;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ध्रुव std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    यदि (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->दत्त == शून्यसूचकः ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_आकार(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_row_आकार(GGML_TYPE_F32, D)) {
        प्रत्यागम असत्य;
    }

    अङ्कः_encoded_ops = cpy_idx - ग्रन्थिः_idx + 1;

    प्रत्यागम सत्य;
}

स्थिर webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ध्रुव uint32_t        बहुलकम्_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_आकार(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_आकार(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     बहुलकम्_flags };

    std::vector<wgpu::Bind_गणः_प्रविष्टिः> entries = { ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_गणः_प्रविष्टिः(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    स्वचालित *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension, wg_x, wg_y);

    प्रत्यागम ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
स्थिर std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           पूर्णाङ्क            ग्रन्थिः_idx,
                                                           पूर्णाङ्क &          अङ्कः_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  ग्रन्थिः  = nodes[ग्रन्थिः_idx];

    यदि (ggml_is_रिक्तानि(ग्रन्थिः)) {
        प्रत्यागम std::nullopt;
    }
    यदि ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        प्रत्यागम std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << ग्रन्थिः << ", " << ggml_op_name(ग्रन्थिः->op) << ")");

    ggml_tensor * src0 = ग्रन्थिः->src[0];
    ggml_tensor * src1 = ग्रन्थिः->src[1];
    ggml_tensor * src2 = ग्रन्थिः->src[2];

    विकल्प (ग्रन्थिः->op) {
            // no-ops
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_RESHAPE:
            प्रत्यागम std::nullopt;
        पक्ष GGML_OP_DUP:
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_CONT:
            प्रत्यागम ggml_webgpu_cpy(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_समुच्चयः:
            प्रत्यागम ggml_webgpu_समुच्चयः(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            प्रत्यागम ggml_webgpu_समुच्चयः_पङ्क्तिकाः(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            प्रत्यागम ggml_webgpu_get_पङ्क्तिकाः(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_MUL_MAT:
            प्रत्यागम ggml_webgpu_mul_mat(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_MUL_MAT_ID:
            प्रत्यागम ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, ग्रन्थिः);
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            प्रत्यागम ggml_webgpu_flash_attn(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः);
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
            प्रत्यागम ggml_webgpu_द्विमान_op(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_सङ्कलय_ID:
            प्रत्यागम ggml_webgpu_सङ्कलय_id(ctx, src0, src1, src2, ग्रन्थिः);
        पक्ष GGML_OP_CONCAT:
            प्रत्यागम ggml_webgpu_concat(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_REPEAT:
            प्रत्यागम ggml_webgpu_repeat(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_RMS_NORM:
            यदि (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, ग्रन्थिः_idx)) {
                अङ्कः_encoded_ops        = 2;
                ggml_tensor * mul_ग्रन्थिः = nodes[ग्रन्थिः_idx + 1];
                प्रत्यागम ggml_webgpu_rms_norm_mul(ctx, src0, ग्रन्थिः, mul_ग्रन्थिः->src[0], mul_ग्रन्थिः->src[1], mul_ग्रन्थिः);
            } अन्यथा {
                प्रत्यागम ggml_webgpu_row_norm(ctx, src0, ग्रन्थिः);
            }
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_L2_NORM:
            प्रत्यागम ggml_webgpu_row_norm(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_ROPE:
            प्रत्यागम ggml_webgpu_rope(ctx, src0, src1, src2, ग्रन्थिः);
        पक्ष GGML_OP_GLU:
            प्रत्यागम ggml_webgpu_glu(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_SCALE:
            प्रत्यागम ggml_webgpu_scale(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_SOFT_MAX:
            प्रत्यागम ggml_webgpu_soft_max(ctx, src0, src1, src2, ग्रन्थिः);
        पक्ष GGML_OP_UNARY:
        पक्ष GGML_OP_CLAMP:
        पक्ष GGML_OP_FILL:
        पक्ष GGML_OP_LOG:
        पक्ष GGML_OP_SQR:
        पक्ष GGML_OP_SQRT:
        पक्ष GGML_OP_SIN:
        पक्ष GGML_OP_COS:
        पक्ष GGML_OP_DIAG:
        पक्ष GGML_OP_TRI:
            प्रत्यागम ggml_webgpu_unary_op(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_SOLVE_TRI:
            प्रत्यागम ggml_webgpu_solve_tri(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_SSM_CONV:
            प्रत्यागम ggml_webgpu_ssm_conv(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_SSM_अवलोकनम्:
            प्रत्यागम ggml_webgpu_ssm_अवलोकनम्(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः->src[5], ग्रन्थिः->src[6],
                                        ग्रन्थिः);
        पक्ष GGML_OP_GATED_DELTA_NET:
            यदि (ggml_webgpu_can_fuse_gdn_cache(cgraph, ग्रन्थिः_idx, अङ्कः_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[ग्रन्थिः_idx + अङ्कः_encoded_ops - 1]->src[1];
                प्रत्यागम ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः->src[5],
                                                   ग्रन्थिः, dst_fuse);
            } अन्यथा {
                प्रत्यागम ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः->src[5],
                                                   ग्रन्थिः, शून्यसूचकः);
            }
        पक्ष GGML_OP_PAD:
            प्रत्यागम ggml_webgpu_pad(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_ARGMAX:
            प्रत्यागम ggml_webgpu_argmax(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_ARGSORT:
        पक्ष GGML_OP_शिखरम्_K:
            // we reuse the same argsort implementation for top_k
            प्रत्यागम ggml_webgpu_argsort(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_CUMSUM:
            प्रत्यागम ggml_webgpu_cumsum(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_योग:
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
            प्रत्यागम ggml_webgpu_योग_पङ्क्तिकाः(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_CONV_2D:
            प्रत्यागम ggml_webgpu_conv_2d(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_CONV_2D_DW:
            प्रत्यागम ggml_webgpu_conv_2d_dw(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_IM2COL:
            प्रत्यागम ggml_webgpu_im2col(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_UPSCALE:
            प्रत्यागम ggml_webgpu_upscale(ctx, src0, ग्रन्थिः);
        स्वभाव:
            प्रत्यागम std::nullopt;
    }
}

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
स्थिर शून्य ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ध्रुव std::vector<std::string> & pipeline_नामानि,
                                                        uint32_t &                       अङ्कः_inflight_batches) {
    यदि (pipeline_नामानि.रिक्तानि()) {
        प्रत्यागम;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Resolve_Query_समुच्चयः(ctx->profile_timestamp_query_समुच्चयः, 0, ctx->profile_timestamp_query_गणनम्,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_बफर_गन्तव्यम्_बफर(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_गणनम् * परिमाण(uint64_t));

    wgpu::Command_बफर profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, अङ्कः_inflight_batches);

    ध्रुव size_t mapped_आकार = ctx->profile_timestamp_query_गणनम् * परिमाण(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_गणनम् == 2 * pipeline_नामानि.size());

    ggml_backend_webgpu_map_बफर(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::Map_बहुलकम्::Read, 0,
                                   mapped_आकार);
    ध्रुव uint64_t * ts_दत्त = (ध्रुव uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_आकार);

    प्रति (size_t i = 0; i < pipeline_नामानि.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ध्रुव द्विगुण elapsed_ms = द्विगुण(ts_दत्त[2 * i + 1] - ts_दत्त[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_नामानि[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#यदि_अन्तः

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
स्थिर शून्य ggml_backend_webgpu_check_समुच्चयः_पङ्क्तिकाः(webgpu_context & ctx, uint32_t & अङ्कः_inflight_batches) {
#यदि_परिभाषितम् GGML_WEBGPU_CHECK_समुच्चयः_पङ्क्तिकाः
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_बफर_गन्तव्यम्_बफर(ctx->समुच्चयः_पङ्क्तिकाः_dev_error_buf, 0, ctx->समुच्चयः_पङ्क्तिकाः_host_error_buf, 0,
                               ctx->समुच्चयः_पङ्क्तिकाः_host_error_buf.Get_आकार());
    wgpu::Command_बफर commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, अङ्कः_inflight_batches);
    ggml_backend_webgpu_map_बफर(ctx->global_ctx, ctx->समुच्चयः_पङ्क्तिकाः_host_error_buf, wgpu::Map_बहुलकम्::Read, 0,
                                   ctx->समुच्चयः_पङ्क्तिकाः_host_error_buf.Get_आकार());
    ध्रुव uint32_t * error_दत्त = (ध्रुव uint32_t *) ctx->समुच्चयः_पङ्क्तिकाः_host_error_buf.GetConstMappedRange();
    यदि (*error_दत्त) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->समुच्चयः_पङ्क्तिकाः_host_error_buf.Unmap();
#अन्यथा
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(अङ्कः_inflight_batches);
#यदि_अन्तः
}

स्थिर ggml_स्थितिः ggml_backend_webgpu_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_योग_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t अङ्कः_batched_kernels  = 0;
    uint32_t अङ्कः_inflight_batches = 0;
    तार्किक     अन्तर्भवति_समुच्चयः_पङ्क्तिकाः    = असत्य;
    पूर्णाङ्क      अङ्कः_encoded_ops      = 1;
    पूर्णाङ्क      ग्रन्थिः_idx             = 0;

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_गणनम् = 0;
    std::vector<std::string> profile_pipeline_नामानि;
#यदि_अन्तः

    ctx->सक्रिय_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    यदि (ctx->batch_compute_passes) {
        ctx->सक्रिय_compute_pass = ctx->सक्रिय_command_encoder.BeginComputePass();
    }

    यावत् (ग्रन्थिः_idx < cgraph->n_nodes) {
        यदि (cgraph->nodes[ग्रन्थिः_idx]->op == GGML_OP_समुच्चयः_पङ्क्तिकाः) {
            अन्तर्भवति_समुच्चयः_पङ्क्तिकाः = सत्य;
        }
        यदि (स्वचालित cmd = ggml_webgpu_encode(ctx, cgraph, ग्रन्थिः_idx, अङ्कः_encoded_ops)) {
            commands.push_back(*cmd);
            अङ्कः_batched_kernels += cmd.value().अङ्कः_kernels;
#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_नामानि.insert(profile_pipeline_नामानि.end(), cmd->pipeline_नामानि.begin(),
                                          cmd->pipeline_नामानि.end());
#यदि_अन्तः
        }

        यदि (अङ्कः_batched_kernels >= ctx->global_ctx->command_submit_batch_आकार) {
            यदि (ctx->सक्रिय_compute_pass) {
                ctx->सक्रिय_compute_pass.अन्तः_3();
            }
            अङ्कः_batched_kernels                = 0;
            wgpu::Command_बफर batch_commands = ctx->सक्रिय_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, अङ्कः_inflight_batches);

            // reset state for next batch
            ctx->सक्रिय_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            यदि (ctx->batch_compute_passes) {
                ctx->सक्रिय_compute_pass = ctx->सक्रिय_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            यदि (ctx->profile_timestamp_query_गणनम् + 2 * ctx->global_ctx->command_submit_batch_आकार >=
                WEBGPU_MAX_PROFILE_QUERY_गणनम्) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_नामानि, अङ्कः_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_गणनम् = 0;
                profile_pipeline_नामानि.clear();
            }
#यदि_अन्तः
        }

        ग्रन्थिः_idx += अङ्कः_encoded_ops;
        अङ्कः_encoded_ops = 1;
    }

    यदि (ctx->सक्रिय_compute_pass) {
        ctx->सक्रिय_compute_pass.अन्तः_3();
        ctx->सक्रिय_compute_pass = शून्यसूचकः;
    }

    यदि (अङ्कः_batched_kernels > 0) {
        wgpu::Command_बफर batch_commands = ctx->सक्रिय_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, अङ्कः_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->सक्रिय_command_encoder = शून्यसूचकः;

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_नामानि, अङ्कः_inflight_batches);
#यदि_अन्तः

    यदि (अन्तर्भवति_समुच्चयः_पङ्क्तिकाः) {
        ggml_backend_webgpu_check_समुच्चयः_पङ्क्तिकाः(ctx, अङ्कः_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_योग_अन्तः(graph_compute, ctx->global_ctx);
    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

संरचना ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    तार्किक                  recorded = असत्य;
};

स्थिर ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    स्वचालित * event_ctx      = नूतन ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    स्वचालित * event   = नूतन ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    प्रत्यागम event;
}

स्थिर शून्य ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_अप्रयुक्तम्(dev);
    विलोपय static_cast<ggml_backend_webgpu_event_context *>(event->context);
    विलोपय event;
}

स्थिर शून्य ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_अप्रयुक्तम्(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    यदि (!event_ctx->recorded) {
        प्रत्यागम;
    }
    wgpu::Wait_स्थितिः स्थितिः =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    यदि (स्थितिः == wgpu::Wait_स्थितिः::Timed_बहिर्गमनम्) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = असत्य;
}

स्थिर शून्य ggml_backend_webgpu_event_लेखः(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.On_Submitted_Work_समाप्तम्(
        wgpu::Callback_बहुलकम्::AllowSpontaneous, [](wgpu::Queue_Work_समाप्तम्_स्थितिः, wgpu::StringView) {});
    event_ctx->recorded = सत्य;
}

स्थिर शून्य ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_अप्रयुक्तम्(backend);
    ggml_backend_webgpu_device_event_synchronize(शून्यसूचकः, event);
}

स्थिर शून्य ggml_backend_webgpu_समुच्चयः_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ध्रुव शून्य *   दत्त,
                                                 size_t         विस्थापनम्,
                                                 size_t         size) {
    GGML_अप्रयुक्तम्(backend);
    स्वचालित * buf_ctx      = (ggml_backend_webgpu_बफर_context *) tensor->बफर->context;
    size_t योग_विस्थापनम् = ggml_webgpu_tensor_विस्थापनम्(tensor) + विस्थापनम्;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_बफर(buf_ctx->बफर, योग_विस्थापनम्, दत्त, (size / 4) * 4);

    यदि (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_आकार = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        प्रति (size_t i = 0; i < remaining_आकार; i++) {
            ((uint8_t *) &val32)[i] = ((ध्रुव uint8_t *) दत्त)[size - remaining_आकार + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, val32,
                                          योग_विस्थापनम् + (size - remaining_आकार), remaining_आकार);
    }
}

स्थिर शून्य ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

स्थिर ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_समुच्चयः_tensor_async,
    /* .get_tensor_async        = */ रिक्त,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ रिक्त,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_लेखः,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ रिक्त,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

स्थिर शून्य ggml_backend_webgpu_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_webgpu_बफर_context * ctx = static_cast<ggml_backend_webgpu_बफर_context *>(बफर->context);
    यदि (ctx != शून्यसूचकः && ctx->बफर != शून्यसूचकः) {
        ctx->बफर.Destroy();
        विलोपय ctx;
    }
}

// Returns the "fake" base pointer.
स्थिर शून्य * ggml_backend_webgpu_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    GGML_अप्रयुक्तम्(बफर);
    प्रत्यागम webgpu_ptr_आधारः;
}

स्थिर शून्य ggml_backend_webgpu_बफर_memset_tensor(ggml_backend_बफर_t बफर,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                विस्थापनम्,
                                                     size_t                size) {
    यदि (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        प्रत्यागम;
    }

    WEBGPU_CPU_PROFILE_योग_START(memset_tensor);

    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << विस्थापनम् << ", " << size << ")");

    size_t योग_विस्थापनम् = ggml_webgpu_tensor_विस्थापनम्(tensor) + विस्थापनम्;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, val32, योग_विस्थापनम्, size);
    WEBGPU_CPU_PROFILE_योग_अन्तः(memset_tensor, buf_ctx->global_ctx);
}

स्थिर शून्य ggml_backend_webgpu_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर,
                                                  ggml_tensor *         tensor,
                                                  ध्रुव शून्य *          दत्त,
                                                  size_t                विस्थापनम्,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_योग_START(समुच्चयः_tensor);
    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << दत्त
                                                              << ", " << विस्थापनम् << ", " << size << ")");

    size_t योग_विस्थापनम् = ggml_webgpu_tensor_विस्थापनम्(tensor) + विस्थापनम्;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    यदि (योग_विस्थापनम् % 4 != 0) {
        size_t lane = योग_विस्थापनम् % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t शीर्ष = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t शीर्ष_val = 0;
        प्रति (size_t i = 0; i < शीर्ष; i++) {
            ((uint8_t *) &शीर्ष_val)[lane + i] = ((ध्रुव uint8_t *) दत्त)[i];
        }
        ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, शीर्ष_val, योग_विस्थापनम्, शीर्ष);

        योग_विस्थापनम् += शीर्ष;
        size -= शीर्ष;
        दत्त = (ध्रुव uint8_t *) दत्त + शीर्ष;
    }

    यदि (size > 0) {
        buf_ctx->global_ctx->queue.Write_बफर(buf_ctx->बफर, योग_विस्थापनम्, दत्त, (size / 4) * 4);
    }

    यदि (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_आकार = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        प्रति (size_t i = 0; i < remaining_आकार; i++) {
            ((uint8_t *) &val32)[i] = ((ध्रुव uint8_t *) दत्त)[size - remaining_आकार + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, val32,
                                          योग_विस्थापनम् + (size - remaining_आकार), remaining_आकार);
    }
    WEBGPU_CPU_PROFILE_योग_अन्तः(समुच्चयः_tensor, buf_ctx->global_ctx);
}

स्थिर शून्य ggml_backend_webgpu_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                  ध्रुव ggml_tensor *   tensor,
                                                  शून्य *                दत्त,
                                                  size_t                विस्थापनम्,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_योग_START(get_tensor);
    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << दत्त
                                                              << ", " << विस्थापनम् << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t योग_विस्थापनम् = ggml_webgpu_tensor_विस्थापनम्(tensor) + विस्थापनम्;

    size_t local_विस्थापनम् = योग_विस्थापनम् % 4;
    यदि (local_विस्थापनम् != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        योग_विस्थापनम् -= local_विस्थापनम्;
    }

    size_t final_आकार = size + local_विस्थापनम्;
    यदि (final_आकार % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_आकार += 4 - (final_आकार % 4);
    }

    std::lock_guard<std::recursive_mutex> तालकम्(buf_ctx->global_ctx->mutex);

    यदि (buf_ctx->global_ctx->get_tensor_staging_buf == शून्यसूचकः ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_आकार() < final_आकार) {
        // Create a new staging buffer if it doesn't exist or is too small
        यदि (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_बफर(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_आकार,
                                  wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_बफर_गन्तव्यम्_बफर(buf_ctx->बफर, योग_विस्थापनम्, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_आकार);
    wgpu::Command_बफर commands = encoder.Finish();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_बफर(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::Map_बहुलकम्::Read, 0, final_आकार);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ध्रुव शून्य * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_आकार);

    // Copy the data from the mapped range to the output buffer
    std::स्मृतिं_प्रतिलिख(दत्त, (ध्रुव शून्य *) ((ध्रुव वर्ण *) mapped_range + local_विस्थापनम्), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_योग_अन्तः(get_tensor, buf_ctx->global_ctx);
}

स्थिर शून्य ggml_backend_webgpu_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << बफर << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_योग_START(clear);
    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;
    ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, value, 0, बफर->size);
    WEBGPU_CPU_PROFILE_योग_अन्तः(clear, buf_ctx->global_ctx);
}

स्थिर ggml_backend_बफर_i ggml_backend_webgpu_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_webgpu_बफर_get_आधारः,
    /* .init_tensor     = */ रिक्त,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_बफर_समुच्चयः_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_बफर_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ रिक्त,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_बफर_clear,
    /* .reset           = */ रिक्त,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    प्रत्यागम ctx->device_name.c_str();
}

स्थिर ggml_backend_बफर_t ggml_backend_webgpu_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                          size_t                     size) {
    स्थिर std::atomic<पूर्णाङ्क> बफर_गणनम्;
    पूर्णाङ्क                     बफर_id = बफर_गणनम्++;
    std::string             buf_name  = "tensor_buf" + std::to_string(बफर_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << बफर_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::बफर_2                         buf;
    ggml_webgpu_create_बफर(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::बफर_Usage::Storage | wgpu::बफर_Usage::CopySrc | wgpu::बफर_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_बफर_context * buf_ctx =
        नूतन ggml_backend_webgpu_बफर_context(buf, buf_name, ctx->webgpu_global_ctx);

    प्रत्यागम ggml_backend_बफर_init(buft, ggml_backend_webgpu_बफर_interface, buf_ctx, size);
}

स्थिर size_t ggml_backend_webgpu_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    प्रत्यागम dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
स्थिर size_t ggml_backend_webgpu_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    प्रत्यागम dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार;
}

स्थिर size_t ggml_backend_webgpu_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft,
                                                             ध्रुव ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    विकल्प (tensor->op) {
        पक्ष GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            भङ्ग;
        पक्ष GGML_OP_शिखरम्_K:
            {
                ध्रुव ggml_tensor * src0 = tensor->src[0];
                यदि (src0) {
                    ध्रुव size_t पूर्णम् = परिमाण(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        पूर्णम् * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                ध्रुव ggml_tensor * Q            = tensor->src[0];
                ध्रुव ggml_tensor * K            = tensor->src[1];
                ध्रुव ggml_tensor * V            = tensor->src[2];
                ध्रुव ggml_tensor * आवरणम्         = tensor->src[3];
                ध्रुव स्वचालित &        capabilities = ctx->webgpu_global_ctx->capabilities;
                यदि (ggml_webgpu_flash_attn_use_vec_मार्गः(ctx->webgpu_global_ctx, Q, K, V)) {
                    ध्रुव तार्किक kv_प्रत्यक्षम् =
                        ggml_webgpu_flash_attn_k_प्रत्यक्षम्(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः) ||
                        ggml_webgpu_flash_attn_v_प्रत्यक्षम्(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः);
                    ध्रुव uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_आकार, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        आवरणम् != शून्यसूचकः, kv_प्रत्यक्षम्);

                    ध्रुव uint32_t vec_nwg_cap = capabilities.min_subgroup_आकार;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ध्रुव size_t   align = capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment;
                    ध्रुव uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    यदि (nwg > 1u) {
                        ध्रुव uint64_t tmp_दत्त_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ध्रुव uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ध्रुव size_t   tmp_आकार_अष्टकानि = ROUNDUP_POW2((tmp_दत्त_elems + tmp_stats_elems) * परिमाण(भिन्न),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_आकार_अष्टकानि + align;
                    } अन्यथा {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    यदि (आवरणम् != शून्यसूचकः) {
                        ध्रुव uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ध्रुव uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ध्रुव uint32_t stride_mask3    = (uint32_t) (आवरणम्->nb[3] / ggml_type_आकार(आवरणम्->type));
                        ध्रुव uint32_t blk_batch_गणनम् = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ध्रुव uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_गणनम्;
                        ध्रुव size_t   blk_आकार_अष्टकानि =
                            ROUNDUP_POW2(blk_elems * परिमाण(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_आकार_अष्टकानि + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT:
            {
                ध्रुव ggml_tensor * src0 = tensor->src[0];
                ध्रुव ggml_tensor * src1 = tensor->src[1];
                तार्किक                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_गुणनफलम्,
                                             ctx->webgpu_global_ctx->vendor);
                यदि (use_mmvq) {
                    ध्रुव size_t q8_src1_आकार = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_आकार +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                ध्रुव ggml_tensor * src0 = tensor->src[0];
                ध्रुव ggml_tensor * src1 = tensor->src[1];
                यदि (src0 && src1) {
                    ध्रुव size_t ids_cache_आकार = परिमाण(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    ध्रुव size_t ids_tiles_आकार = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_आकार + ids_tiles_आकार +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            भङ्ग;
        स्वभाव:
            भङ्ग;
    }
    प्रत्यागम res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    प्रत्यागम ctx->device_name.c_str();
}

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    प्रत्यागम ctx->device_desc.c_str();
}

स्थिर शून्य ggml_backend_webgpu_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_बफर_आकार = ctx->webgpu_global_ctx->capabilities.limits.max_बफर_आकार_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#यदि UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_आकार = static_cast<uint64_t>(UINTPTR_MAX);
    यदि (max_बफर_आकार > max_ptr_आकार) {
        max_बफर_आकार = max_ptr_आकार;
    }
#यदि_अन्तः
    *मुक्त  = static_cast<size_t>(max_बफर_आकार);
    *योग = static_cast<size_t>(max_बफर_आकार);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_अप्रयुक्तम्(dev);
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_GPU;
}

स्थिर शून्य ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);
    props->caps = {
        /* .async                 = */ असत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ असत्य,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_guid_t ggml_backend_webgpu_guid(शून्य) {
    स्थिर ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    प्रत्यागम &guid;
}

स्थिर शून्य ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_तन्तवः = ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup *
                         ctx->capabilities.limits.max_Compute_Workgroups_प्रत्येकम्_Dimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_अष्टकानि_प्रत्येकम्_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_बफर_Binding_आकार, max_तन्तवः);
    std::vector<wgpu::Constant_प्रविष्टिः> constants(2);
    constants[0].कुञ्चिका     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
    constants[1].कुञ्चिका     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_अष्टकानि_प्रत्येकम्_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

स्थिर शून्य ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#यदि !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ध्रुव वर्ण * ध्रुव          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_गणनम् = 2;
    options.अग्र_अन्तर्_Chain                   = &adapterTogglesDesc;
#यदि_अन्तः

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::Callback_बहुलकम्::AllowSpontaneous,
            [&adapter](wgpu::Request_Adapter_स्थितिः स्थितिः, wgpu::Adapter _adapter, wgpu::StringView सन्देशः) {
                यदि (स्थितिः != wgpu::Request_Adapter_स्थितिः::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(सन्देशः).c_str());
                    प्रत्यागम;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

स्थिर शून्य create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != शून्यसूचकः);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#यदि !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    यदि (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.अग्र_अन्तर्_Chain = &subgroup_matrix_configs;
    }
#यदि_अन्तः
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_आकार = ggml_backend_webgpu_get_command_submit_batch_आकार();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_गुणनफलम् = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8_Integer_Dot_गुणनफलम्);

    तार्किक valid_subgroup_matrix_config = असत्य;
#यदि !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    यदि (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        प्रति (size_t i = 0; i < subgroup_matrix_configs.config_गणनम्; i++) {
            ध्रुव wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            यदि (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.परिणाम_Component_Type == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = सत्य;
                भङ्ग;
            }
        }
    }
#यदि_अन्तः
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_आकार = info.subgroup_Min_आकार;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_आकार = info.subgroup_Max_आकार;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#यदि !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    यदि (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#यदि_अन्तः

    यदि (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#यदि_अन्तः

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.दत्त();
    dev_desc.required_Feature_गणनम् = required_features.size();
    dev_desc.समुच्चयः_Device_नष्टम्_Callback(
        wgpu::Callback_बहुलकम्::AllowSpontaneous,
        [](ध्रुव wgpu::Device & device, wgpu::Device_नष्टम्_Reason reason, wgpu::StringView सन्देशः) {
            यदि (reason == wgpu::Device_नष्टम्_Reason::Destroyed) {
                प्रत्यागम;
            }
            GGML_अप्रयुक्तम्(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<पूर्णाङ्क>(reason),
                           std::string(सन्देशः).c_str());
        });
    dev_desc.समुच्चयः_Uncaptured_Error_Callback(
        [](ध्रुव wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView सन्देशः) {
            GGML_अप्रयुक्तम्(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<पूर्णाङ्क>(reason),
                       std::string(सन्देशः).c_str());
        });

#यदि !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    ध्रुव वर्ण * ध्रुव          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    ध्रुव वर्ण * ध्रुव          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_गणनम्  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_गणनम् = 1;

    dev_desc.अग्र_अन्तर्_Chain = &deviceTogglesDesc;
#यदि_अन्तः

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::Callback_बहुलकम्::AllowSpontaneous,
            [ctx](wgpu::Request_Device_स्थितिः स्थितिः, wgpu::Device device, wgpu::StringView सन्देशः) {
                यदि (स्थितिः != wgpu::Request_Device_स्थितिः::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(सन्देशः).c_str());
                    प्रत्यागम;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != शून्यसूचकः);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_बफर(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_आकार_अष्टकानि, wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.GetQueue();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

स्थिर webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_आकार_अष्टकानि,
        webgpu_ctx->global_ctx->command_submit_batch_आकार + WEBGPU_अङ्कः_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_एकरूपम्_बफर_विस्थापनम्_Alignment);
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->समुच्चयः_पङ्क्तिकाः_dev_error_buf,
                              WEBGPU_समुच्चयः_पङ्क्तिकाः_ERROR_BUF_आकार_अष्टकानि,
                              wgpu::बफर_Usage::Storage | wgpu::बफर_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->समुच्चयः_पङ्क्तिकाः_host_error_buf,
                              WEBGPU_समुच्चयः_पङ्क्तिकाः_ERROR_BUF_आकार_अष्टकानि,
                              wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "set_rows_host_error_buf");

#यदि_परिभाषितम् GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = असत्य;
    ggml_webgpu_create_बफर(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_आकार_अष्टकानि,
        wgpu::बफर_Usage::QueryResolve | wgpu::बफर_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_आकार_अष्टकानि,
                              wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::Query_समुच्चयः_Descriptor query_समुच्चयः_desc = {};
    query_समुच्चयः_desc.type                     = wgpu::QueryType::Timestamp;
    query_समुच्चयः_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_गणनम्;
    webgpu_ctx->profile_timestamp_query_समुच्चयः = webgpu_ctx->global_ctx->device.Create_Query_समुच्चयः(&query_समुच्चयः_desc);
#यदि_अन्तः

#यदि_परिभाषितम् GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * परिमाण(uint32_t),
                              wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * परिमाण(uint32_t),
                              wgpu::बफर_Usage::Storage | wgpu::बफर_Usage::CopySrc, "debug_dev_buf");
#यदि_अन्तः
    प्रत्यागम webgpu_ctx;
}

स्थिर ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    GGML_अप्रयुक्तम्(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    स्वचालित * backend_ctx      = नूतन ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    स्वचालित * backend = नूतन ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    प्रत्यागम backend;
}

स्थिर ggml_backend_बफर_type_t ggml_backend_webgpu_device_get_बफर_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    स्थिर संरचना ggml_backend_बफर_type ggml_backend_webgpu_बफर_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_बफर_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_बफर_type_alloc_बफर,
                        /* .alloc_buffer_n      = */ रिक्त,
                        /* .get_alignment       = */ ggml_backend_webgpu_बफर_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_बफर_type_get_max_आकार,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_बफर_type_get_alloc_आकार,
                        /* .get_alloc_size_n    = */ रिक्त,
                        /* .is_host             = */ रिक्त,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ रिक्त
    };

    प्रत्यागम &ggml_backend_webgpu_बफर_type;
}

स्थिर तार्किक ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    GGML_अप्रयुक्तम्(dev);
    प्रत्यागम buft->iface.get_name == ggml_backend_webgpu_बफर_type_get_name;
}

स्थिर तार्किक ggml_webgpu_supported_qtype(ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

स्थिर तार्किक ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    यदि (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार ||
        (src0 != शून्यसूचकः &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार) ||
        (src1 != शून्यसूचकः &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार)) {
        प्रत्यागम असत्य;
    }

    तार्किक supports_op = असत्य;
    विकल्प (op->op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_RESHAPE:
            supports_op = सत्य;
            भङ्ग;
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_सङ्कलय_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            भङ्ग;
        पक्ष GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            भङ्ग;
        पक्ष GGML_OP_DUP:
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            भङ्ग;
        पक्ष GGML_OP_समुच्चयः:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            भङ्ग;
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            यदि (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } अन्यथा यदि (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT:
            {
                विकल्प (src1->type) {
                    पक्ष GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        भङ्ग;
                    पक्ष GGML_TYPE_F32:
                        विकल्प (src0->type) {
                            पक्ष GGML_TYPE_F32:
                            पक्ष GGML_TYPE_F16:
                            पक्ष GGML_TYPE_BF16:
                            पक्ष GGML_TYPE_Q1_0:
                            पक्ष GGML_TYPE_Q2_0:
                            पक्ष GGML_TYPE_Q4_0:
                            पक्ष GGML_TYPE_Q4_1:
                            पक्ष GGML_TYPE_Q5_0:
                            पक्ष GGML_TYPE_Q5_1:
                            पक्ष GGML_TYPE_Q8_0:
                            पक्ष GGML_TYPE_Q2_K:
                            पक्ष GGML_TYPE_Q3_K:
                            पक्ष GGML_TYPE_Q4_K:
                            पक्ष GGML_TYPE_Q5_K:
                            पक्ष GGML_TYPE_Q6_K:
                            पक्ष GGML_TYPE_IQ2_XXS:
                            पक्ष GGML_TYPE_IQ2_XS:
                            पक्ष GGML_TYPE_IQ2_S:
                            पक्ष GGML_TYPE_IQ3_XXS:
                            पक्ष GGML_TYPE_IQ3_S:
                            पक्ष GGML_TYPE_IQ1_S:
                            पक्ष GGML_TYPE_IQ1_M:
                            पक्ष GGML_TYPE_IQ4_NL:
                            पक्ष GGML_TYPE_IQ4_XS:
                            पक्ष GGML_TYPE_MXFP4:
                            पक्ष GGML_TYPE_NVFP4:
                                supports_op = सत्य;
                                भङ्ग;
                            स्वभाव:
                                भङ्ग;
                        }
                    स्वभाव:
                        भङ्ग;
                }
                भङ्ग;
            }
        पक्ष GGML_OP_MUL_MAT_ID:
            विकल्प (src1->type) {
                पक्ष GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    भङ्ग;
                पक्ष GGML_TYPE_F32:
                    विकल्प (src0->type) {
                        पक्ष GGML_TYPE_F32:
                        पक्ष GGML_TYPE_F16:
                        पक्ष GGML_TYPE_BF16:
                        पक्ष GGML_TYPE_Q1_0:
                        पक्ष GGML_TYPE_Q4_0:
                        पक्ष GGML_TYPE_Q4_1:
                        पक्ष GGML_TYPE_Q5_0:
                        पक्ष GGML_TYPE_Q5_1:
                        पक्ष GGML_TYPE_Q8_0:
                        पक्ष GGML_TYPE_Q2_K:
                        पक्ष GGML_TYPE_Q3_K:
                        पक्ष GGML_TYPE_Q4_K:
                        पक्ष GGML_TYPE_Q5_K:
                        पक्ष GGML_TYPE_Q6_K:
                        पक्ष GGML_TYPE_IQ1_S:
                        पक्ष GGML_TYPE_IQ1_M:
                        पक्ष GGML_TYPE_IQ2_XXS:
                        पक्ष GGML_TYPE_IQ2_XS:
                        पक्ष GGML_TYPE_IQ2_S:
                        पक्ष GGML_TYPE_IQ3_XXS:
                        पक्ष GGML_TYPE_IQ3_S:
                        पक्ष GGML_TYPE_IQ4_NL:
                        पक्ष GGML_TYPE_IQ4_XS:
                        पक्ष GGML_TYPE_MXFP4:
                        पक्ष GGML_TYPE_NVFP4:
                            supports_op = सत्य;
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }
                    भङ्ग;
                स्वभाव:
                    भङ्ग;
            }
            यदि (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                यदि (!supports_op) {
                    भङ्ग;
                }
                यदि (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = असत्य;
                    भङ्ग;
                }
                ध्रुव स्वचालित & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ध्रुव size_t storage_विस्थापनम्_alignment = capabilities.limits.min_Storage_बफर_विस्थापनम्_Alignment;

                // subgroup matrix path requirements
                ध्रुव तार्किक use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_मार्गः(
                    capabilities.supports_subgroup_matrix, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ध्रुव तार्किक float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_विस्थापनम्_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_विस्थापनम्_alignment));
                ध्रुव uint32_t k_tile_शीर्ष_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः :
                                                       (uint32_t) ggml_blck_आकार(src1->type);
                ध्रुव uint32_t v_tile_शीर्ष_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_विस्तारः :
                                                       (uint32_t) ggml_blck_आकार(src2->type);
                ध्रुव तार्किक     tile_kv_शीर्ष_dims_aligned =
                    src0->ne[0] % k_tile_शीर्ष_align == 0 && src2->ne[0] % v_tile_शीर्ष_align == 0;
                ध्रुव तार्किक tile_can_dispatch_all_q_पङ्क्तिकाः =
                    capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_आकार;
                ध्रुव तार्किक use_tile = !use_subgroup_matrix && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_शीर्ष_dims_aligned && tile_can_dispatch_all_q_पङ्क्तिकाः;

                यदि (!use_subgroup_matrix && !use_tile) {
                    supports_op = असत्य;
                    भङ्ग;
                }
                ध्रुव uint32_t q_tile =
                    use_subgroup_matrix ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ध्रुव uint32_t kv_granularity = use_subgroup_matrix ? capabilities.sg_mat_n : 1u;
                ध्रुव तार्किक     kv_प्रत्यक्षम् = use_subgroup_matrix ?
                                               ggml_webgpu_flash_attn_k_प्रत्यक्षम्(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_प्रत्यक्षम्(src0, src2, capabilities.sg_mat_k) :
                                               असत्य;
                ध्रुव uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_आकार, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != शून्यसूचकः, kv_प्रत्यक्षम्);
                supports_op = max_kv_tile > 0;
                भङ्ग;
            }
        पक्ष GGML_OP_RMS_NORM:
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_पङ्क्तिकाः(src0);
            भङ्ग;
        पक्ष GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            भङ्ग;
        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(op)) {
                पक्ष GGML_GLU_OP_REGLU:
                पक्ष GGML_GLU_OP_GEGLU:
                पक्ष GGML_GLU_OP_SWIGLU:
                पक्ष GGML_GLU_OP_GEGLU_ERF:
                पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    भङ्ग;
                पक्ष GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    भङ्ग;
                स्वभाव:
                    भङ्ग;
            }
            भङ्ग;
        पक्ष GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_UNARY:
            {
                ध्रुव ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                विकल्प (UNARY_OP) {
                    पक्ष GGML_UNARY_OP_ABS:
                    पक्ष GGML_UNARY_OP_SGN:
                    पक्ष GGML_UNARY_OP_NEG:
                    पक्ष GGML_UNARY_OP_चरणः:
                    पक्ष GGML_UNARY_OP_TANH:
                    पक्ष GGML_UNARY_OP_ELU:
                    पक्ष GGML_UNARY_OP_RELU:
                    पक्ष GGML_UNARY_OP_SIGMOID:
                    पक्ष GGML_UNARY_OP_GELU:
                    पक्ष GGML_UNARY_OP_GELU_शीघ्र:
                    पक्ष GGML_UNARY_OP_SILU:
                    पक्ष GGML_UNARY_OP_HARDSWISH:
                    पक्ष GGML_UNARY_OP_HARDSIGMOID:
                    पक्ष GGML_UNARY_OP_EXP:
                    पक्ष GGML_UNARY_OP_GELU_ERF:
                    पक्ष GGML_UNARY_OP_SOFTPLUS:
                    पक्ष GGML_UNARY_OP_EXPM1:
                    पक्ष GGML_UNARY_OP_FLOOR:
                    पक्ष GGML_UNARY_OP_CEIL:
                    पक्ष GGML_UNARY_OP_आवर्तनम्:
                    पक्ष GGML_UNARY_OP_TRUNC:
                    पक्ष GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        भङ्ग;
                    स्वभाव:
                        भङ्ग;
                }
            }
            भङ्ग;
        पक्ष GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            भङ्ग;
        पक्ष GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            भङ्ग;
        पक्ष GGML_OP_IM2COL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            भङ्ग;
        पक्ष GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_SSM_अवलोकनम्:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
            भङ्ग;
        पक्ष GGML_OP_GATED_DELTA_NET:
            {
                ध्रुव uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_प्रत्येकम्_Workgroup;
            }
            भङ्ग;
        पक्ष GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            भङ्ग;
        पक्ष GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_पङ्क्तिकाः(src0);
            भङ्ग;
        पक्ष GGML_OP_शिखरम्_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_पङ्क्तिकाः(src0);
            भङ्ग;
        पक्ष GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            भङ्ग;
        पक्ष GGML_OP_योग:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            भङ्ग;
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_पङ्क्तिकाः(src0);
            भङ्ग;
        पक्ष GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            भङ्ग;
        स्वभाव:
            भङ्ग;
    }
    यदि (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार ||
        (src0 != शून्यसूचकः &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार) ||
        (src1 != शून्यसूचकः &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार) ||
        (src2 != शून्यसूचकः &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार)) {
        supports_op = असत्य;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    यदि (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } अन्यथा {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    प्रत्यागम supports_op;
}

स्थिर संरचना ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_स्मृतिः,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_बफर_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ रिक्त,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ रिक्त,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    प्रत्यागम ctx->name;
}

स्थिर size_t ggml_backend_webgpu_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    प्रत्यागम ctx->device_गणनम्;
}

// Only one device is supported for now
स्थिर ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_योग_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    स्थिर ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    स्थिर ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_योग_अन्तः(reg_get_device, reg_ctx->webgpu_global_ctx);
    प्रत्यागम &device;
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_गणनम्,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ रिक्त,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    स्थिर ggml_backend_webgpu_reg_context * ctx = नूतन ggml_backend_webgpu_reg_context();

    स्थिर ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_गणनम् = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    यदि (ctx->webgpu_global_ctx != शून्यसूचकः && ctx->webgpu_global_ctx->instance != शून्यसूचकः) {
        प्रत्यागम &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.दत्त();
    instance_descriptor.required_Feature_गणनम्                 = instance_features.size();

#यदि !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ध्रुव वर्ण * ध्रुव          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_गणनम् = 1;
    instance_descriptor.अग्र_अन्तर्_Chain        = &instanceTogglesDesc;
#यदि_अन्तः

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(नूतन webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    यदि (ctx->webgpu_global_ctx->instance != शून्यसूचकः) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    यदि (adapter != शून्यसूचकः && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#यदि !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#यदि_अन्तः
    ) {
        ctx->device_गणनम् = 1;
    }

    प्रत्यागम &reg;
}

ggml_backend_t ggml_backend_webgpu_init(शून्य) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    यदि (ggml_backend_reg_dev_गणनम्(reg) == 0) {
        प्रत्यागम शून्यसूचकः;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    प्रत्यागम ggml_backend_webgpu_backend_init(dev, शून्यसूचकः);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
