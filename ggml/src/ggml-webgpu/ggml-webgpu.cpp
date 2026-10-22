/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#समावेश "ggml-webgpu.h"

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-webgpu-shader-lib.hpp"
#समावेश "ggml.h"

#अगर_घोषित __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#पूर्ण_अगर

#समावेश <webgpu/webgpu_cpp.h>

#समावेश <atomic>
#समावेश <cstdint>
#समावेश <cstring>
#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#पूर्ण_अगर
#अगर defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#पूर्ण_अगर
#समावेश <memory>
#समावेश <mutex>
#समावेश <optional>
#समावेश <string>
#समावेश <utility>
#समावेश <vector>

#घोषणा ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#घोषणा CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
अटल अंतरभूत व्योम compute_2d_workgroups(uint32_t योग_wg, uint32_t max_per_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(योग_wg, max_per_dim));
    wg_x = CEIL_DIV(योग_wg, wg_y);
}

अटल अंतरभूत uint32_t ggml_webgpu_u32_from_f32(भग्न value) {
    uint32_t bits;
    स_नकल(&bits, &value, माप(bits));
    वापस bits;
}

#अगर_घोषित GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(संदेश)  std::अ_बाहर << संदेश << std::प_पूर्ण
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#अन्यथा
#    define WEBGPU_LOG_DEBUG(संदेश) ((व्योम) 0)
#पूर्ण_अगर  // GGML_WEBGPU_DEBUG

#अगर_घोषित GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_योग_START(id) स्वतः cpu_योग_start##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_योग_END(id, ctx)                                                         \
        स्वतः   cpu_योग_end##id = std::chrono::high_resolution_clock::now();                            \
        द्विगुन cpu_योग_time##id =                                                                      \
            std::chrono::duration<द्विगुन, std::milli>(cpu_योग_end##id - cpu_योग_start##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_योग_time##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) स्वतः cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)                                                          \
        स्वतः   cpu_detail_end_##id = std::chrono::high_resolution_clock::now();                             \
        द्विगुन cpu_detail_time_##id =                                                                       \
            std::chrono::duration<द्विगुन, std::milli>(cpu_detail_end_##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#अन्यथा
#    define WEBGPU_CPU_PROFILE_योग_START(id)
#    define WEBGPU_CPU_PROFILE_योग_END(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)
#पूर्ण_अगर  // GGML_WEBGPU_CPU_PROFILE

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_गिनती        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_आकार_BYTES (WEBGPU_MAX_PROFILE_QUERY_गिनती * माप(uint64_t))
#पूर्ण_अगर

/* Constants */

#घोषणा WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_आकार 64u
#घोषणा WEBGPU_अंक_PARAM_SLOT_SAFETY_MARGIN      10u
#घोषणा WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#घोषणा WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#घोषणा WEBGPU_PARAMS_BUF_आकार_BYTES             128  // enough for 32 parameters
#घोषणा WEBGPU_SET_ROWS_ERROR_BUF_आकार_BYTES     4
#घोषणा WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
अटल व्योम * स्थिर webgpu_ptr_base = (व्योम *) (uintptr_t) 0x1000;  // NOLINT

अटल माप_प्रकार ggml_webgpu_tensor_offset(स्थिर ggml_tensor * tensor) {
    स्थिर ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    वापस (माप_प्रकार) ((uintptr_t) base_tensor->तथ्य - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
अटल व्योम ggml_webgpu_create_बफर(wgpu::Device &    device,
                                      wgpu::बफर_2 &    बफर,
                                      माप_प्रकार            size,
                                      wgpu::बफर_Usage usage,
                                      स्थिर अक्षर *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
काष्ठा webgpu_param_arena {
    wgpu::बफर_2 बफर;
    माप_प्रकार       slot_stride = 0;
    माप_प्रकार       slot_आकार   = 0;
    uint32_t     slot_गिनती  = 0;
    uint32_t     अग्र_slot   = 0;

    व्योम init(wgpu::Device device, माप_प्रकार slot_आकार, uint32_t slot_गिनती, माप_प्रकार alignment) {
        यह->slot_stride = ROUNDUP_POW2(slot_आकार, alignment);
        यह->slot_आकार   = slot_आकार;
        यह->slot_गिनती  = slot_गिनती;
        यह->अग्र_slot   = 0;

        ggml_webgpu_create_बफर(device, बफर, यह->slot_stride * slot_गिनती,
                                  wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    माप_प्रकार alloc_slot(माप_प्रकार size) {
        GGML_ASSERT(size <= slot_आकार);
        अगर (अग्र_slot >= slot_गिनती) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        वापस slot_stride * अग्र_slot++;
    }

    व्योम reset() { अग्र_slot = 0; }

    व्योम cleanup() {
        अगर (बफर) {
            बफर.Destroy();
            बफर = nullptr;
        }
    }

    ~webgpu_param_arena() { यह->cleanup(); }
};

काष्ठा webgpu_encoded_op {
    uint32_t अंक_kernels = 0;
#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_names;
#पूर्ण_अगर
};

काष्ठा webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

काष्ठा webgpu_capabilities {
    wgpu::Limits limits;
    bool         supports_subgroups       = false;
    bool         supports_subgroup_matrix = false;
    bool         supports_dot_product     = false;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_आकार     = 0;
    uint32_t min_subgroup_आकार = 0;
    uint32_t max_subgroup_आकार = 0;
    माप_प्रकार   memset_bytes_per_thread;
};

// Stores global webgpu members
काष्ठा webgpu_global_context_struct {
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
#अगर_घोषित GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, द्विगुन> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, द्विगुन> cpu_detail_ms;
#पूर्ण_अगर

#अगर_घोषित GGML_WEBGPU_DEBUG
    wgpu::बफर_2 debug_host_buf;
    wgpu::बफर_2 debug_dev_buf;
#पूर्ण_अगर

    ~webgpu_global_context_struct() {
        अगर (यह->get_tensor_staging_buf) {
            यह->get_tensor_staging_buf.Destroy();
            यह->get_tensor_staging_buf = nullptr;
        }
        अगर (यह->memset_params_buf) {
            यह->memset_params_buf.Destroy();
            यह->memset_params_buf = nullptr;
        }
#अगर_घोषित GGML_WEBGPU_DEBUG
        अगर (यह->debug_host_buf) {
            यह->debug_host_buf.Destroy();
            यह->debug_host_buf = nullptr;
        }
        अगर (यह->debug_dev_buf) {
            यह->debug_dev_buf.Destroy();
            यह->debug_dev_buf = nullptr;
        }
#पूर्ण_अगर
    }
};

प्रकार std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
काष्ठा webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::बफर_2             set_rows_dev_error_buf;
    wgpu::बफर_2             set_rows_host_error_buf;
    wgpu::CommandEncoder     active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;
    bool                     batch_compute_passes = true;

    माप_प्रकार memset_bytes_per_thread;

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, द्विगुन> shader_gpu_time_ms;
    wgpu::बफर_2                            profile_timestamp_dev_buf;
    wgpu::बफर_2                            profile_timestamp_host_buf;
    wgpu::QuerySet                          profile_timestamp_query_set;
    uint32_t                                profile_timestamp_query_गिनती = 0;
#पूर्ण_अगर

    ~webgpu_context_struct() {
#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
        अगर (यह->profile_timestamp_host_buf) {
            यह->profile_timestamp_host_buf.Destroy();
            यह->profile_timestamp_host_buf = nullptr;
        }
        अगर (यह->profile_timestamp_dev_buf) {
            यह->profile_timestamp_dev_buf.Destroy();
            यह->profile_timestamp_dev_buf = nullptr;
        }
        अगर (यह->profile_timestamp_query_set) {
            यह->profile_timestamp_query_set.Destroy();
            यह->profile_timestamp_query_set = nullptr;
        }
#पूर्ण_अगर
        अगर (यह->set_rows_host_error_buf) {
            यह->set_rows_host_error_buf.Destroy();
            यह->set_rows_host_error_buf = nullptr;
        }
        अगर (यह->set_rows_dev_error_buf) {
            यह->set_rows_dev_error_buf.Destroy();
            यह->set_rows_dev_error_buf = nullptr;
        }
    }
};

प्रकार std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
काष्ठा ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    माप_प्रकार                device_गिनती;
    स्थिर अक्षर *          name;
};

// Per-device struct for the global logical device interface
काष्ठा ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
काष्ठा ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
काष्ठा ggml_backend_webgpu_बफर_context {
    wgpu::बफर_2          बफर;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_बफर_context(wgpu::बफर_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        बफर(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

अटल webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   स्थिर अक्षर *                             shader_code,
                                                   स्थिर अक्षर *                             label,
                                                   स्थिर std::vector<wgpu::ConstantEntry> & constants = {}) {
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.अग्र_In_Chain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = nullptr;  // nullptr means auto layout
    अगर (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.तथ्य();
        pipeline_desc.compute.constant_गिनती = constants.size();
    }
    वापस { device.CreateComputePipeline(&pipeline_desc), label };
}

अटल व्योम ggml_webgpu_create_बफर(wgpu::Device &    device,
                                      wgpu::बफर_2 &    बफर,
                                      माप_प्रकार            size,
                                      wgpu::बफर_Usage usage,
                                      स्थिर अक्षर *      label) {
    wgpu::बफर_Descriptor बफर_desc;
    बफर_desc.size             = size;
    बफर_desc.usage            = usage;
    बफर_desc.label            = label;
    बफर_desc.mappedAtCreation = false;

    // TODO: error handling
    बफर = device.Create_बफर(&बफर_desc);
}

अटल wgpu::बफर_2 ggml_webgpu_tensor_buf(स्थिर ggml_tensor * tensor) {
    ggml_backend_webgpu_बफर_context * ctx = (ggml_backend_webgpu_बफर_context *) tensor->बफर->context;
    वापस ctx->बफर;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
अटल माप_प्रकार ggml_webgpu_tensor_align_offset(स्थिर ggml_tensor * t, माप_प्रकार alignment) {
    स्थिर माप_प्रकार offset    = ggml_webgpu_tensor_offset(t);
    स्थिर माप_प्रकार type_आकार = ggml_type_आकार(t->type);
    माप_प्रकार       aligned   = offset & ~(alignment - 1);
    जबतक ((offset - aligned) % type_आकार != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    वापस aligned;
}

अटल माप_प्रकार ggml_webgpu_tensor_misalignment(स्थिर ggml_tensor * t, माप_प्रकार alignment) {
    वापस ggml_webgpu_tensor_offset(t) - ggml_webgpu_tensor_align_offset(t, alignment);
}

अटल माप_प्रकार ggml_webgpu_tensor_misalignment(webgpu_context & ctx, स्थिर ggml_tensor * t) {
    वापस ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment);
}

अटल माप_प्रकार ggml_webgpu_tensor_align_offset(webgpu_context & ctx, स्थिर ggml_tensor * t) {
    वापस ggml_webgpu_tensor_align_offset(t, ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment);
}

अटल माप_प्रकार ggml_webgpu_tensor_binding_आकार(स्थिर ggml_tensor * t, माप_प्रकार alignment) {
    वापस ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

अटल माप_प्रकार ggml_webgpu_tensor_binding_आकार(webgpu_context & ctx, स्थिर ggml_tensor * t) {
    वापस ggml_webgpu_tensor_binding_आकार(t, ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment);
}

अटल bool ggml_webgpu_tensor_binding_overlap(स्थिर webgpu_global_context & global_ctx,
                                               स्थिर ggml_tensor *           a,
                                               स्थिर ggml_tensor *           b) {
    अगर (a->बफर != b->बफर) {
        वापस false;
    }

    स्थिर माप_प्रकार alignment = global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment;
    स्थिर माप_प्रकार a_offset  = ggml_webgpu_tensor_align_offset(a, alignment);
    स्थिर माप_प्रकार b_offset  = ggml_webgpu_tensor_align_offset(b, alignment);
    वापस a_offset < b_offset + ggml_webgpu_tensor_binding_आकार(b, alignment) &&
           b_offset < a_offset + ggml_webgpu_tensor_binding_आकार(a, alignment);
}

अटल bool ggml_webgpu_tensor_binding_overlap_range(स्थिर webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_बफर_t         बफर,
                                                     माप_प्रकार                        offset,
                                                     माप_प्रकार                        size) {
    अगर (tensor->बफर != बफर) {
        वापस false;
    }

    स्थिर माप_प्रकार alignment     = global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment;
    स्थिर माप_प्रकार tensor_offset = ggml_webgpu_tensor_align_offset(tensor, alignment);
    वापस tensor_offset < offset + size && offset < tensor_offset + ggml_webgpu_tensor_binding_आकार(tensor, alignment);
}

काष्ठा ggml_webgpu_merged_binding_range {
    माप_प्रकार offset;
    माप_प्रकार size;
};

अटल ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    माप_प्रकार merged_offset = SIZE_MAX;
    माप_प्रकार merged_end    = 0;

    क्रम (ggml_tensor * tensor : tensors) {
        स्थिर माप_प्रकार bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
        स्थिर माप_प्रकार bind_end    = bind_offset + ggml_webgpu_tensor_binding_आकार(ctx, tensor);

        merged_offset = std::min(merged_offset, bind_offset);
        merged_end    = std::max(merged_end, bind_end);
    }

    वापस { merged_offset, merged_end - merged_offset };
}

अटल uint32_t ggml_webgpu_tensor_merged_element_offset(स्थिर ggml_tensor *                      tensor,
                                                         स्थिर ggml_webgpu_merged_binding_range & merged_range) {
    वापस (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_आकार(tensor->type));
}

अटल wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
                                                              wgpu::बफर_2 बफर,
                                                              uint64_t     offset,
                                                              uint64_t     size) {
    wgpu::BindGroupEntry entry = {};
    entry.binding              = binding;
    entry.बफर               = std::move(बफर);
    entry.offset               = offset;
    entry.size                 = size;
    वापस entry;
}

अटल wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    वापस ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_offset(ctx, tensor),
                                             ggml_webgpu_tensor_binding_आकार(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

ढाँचा <typename T>
अटल व्योम ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T                callback_status,
                                                  T                success_status,
                                                  स्थिर अक्षर *     wait_name,
                                                  स्थिर अक्षर *     failure_name,
                                                  स्थिर अक्षर *     callback_message) {
    अगर (wait_status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    अगर (wait_status == wgpu::WaitStatus::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    अगर (callback_status != success_status) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<पूर्णांक>(callback_status),
                   callback_message);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
अटल uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    वापस UINT32_MAX;
}

अटल uint32_t ggml_backend_webgpu_get_command_submit_batch_आकार() {
    वापस WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_आकार;
}

अटल व्योम ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::QueueWorkDoneStatus callback_status = wgpu::QueueWorkDoneStatus::Error;
    std::string               callback_message;

    स्थिर wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        ctx->queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [&callback_status, &callback_message](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                callback_status  = status;
                callback_message = std::string(message);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::QueueWorkDoneStatus::Success,
                                          "Queue wait", "Queue work", callback_message.c_str());
}

अटल व्योम ggml_backend_webgpu_map_बफर(webgpu_global_context & ctx,
                                           wgpu::बफर_2 &          बफर,
                                           wgpu::MapMode           mode,
                                           माप_प्रकार                  offset,
                                           माप_प्रकार                  size) {
    wgpu::MapAsyncStatus callback_status = wgpu::MapAsyncStatus::Error;
    std::string          callback_message;

    स्थिर wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        बफर.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
                        [&callback_status, &callback_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                            callback_status  = status;
                            callback_message = std::string(message);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::MapAsyncStatus::Success,
                                          "Buffer map wait", "Buffer map", callback_message.c_str());
}

अटल व्योम ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                स्थिर wgpu::Command_बफर commands,
                                                uint32_t &                अंक_inflight_batches) {
    अगर (अंक_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        अंक_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    अंक_inflight_batches++;
}

#अगर_घोषित GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
अटल व्योम ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_बफर_To_बफर(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_आकार());
    wgpu::Command_बफर commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_बफर(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.Get_आकार());
    स्थिर भग्न * debug_तथ्य = (स्थिर भग्न *) ctx->debug_host_buf.GetConstMappedRange();
    std::अ_बाहर << "debug[0]: " << debug_तथ्य[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#पूर्ण_अगर

अटल webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         स्थिर std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            परिणाम = {};
    std::vector<wgpu::BindGroup> bind_groups;
    std::vector<माप_प्रकार>          param_offsets;
    परिणाम.अंक_kernels = dispatches.size();

    क्रम (माप_प्रकार i = 0; i < dispatches.size(); i++) {
        स्थिर webgpu_dispatch_desc & dispatch     = dispatches[i];
        स्थिर माप_प्रकार                 param_आकार   = dispatch.params.size() * माप(uint32_t);
        स्थिर माप_प्रकार                 param_offset = ctx->param_arena.alloc_slot(param_आकार);

        std::vector<wgpu::BindGroupEntry> entries            = dispatch.bind_group_entries;
        uint32_t                          params_binding_अंक = entries.size();
        entries.push_back(ggml_webgpu_make_bind_group_entry(params_binding_अंक, ctx->param_arena.बफर, param_offset,
                                                            ctx->param_arena.slot_आकार));

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = dispatch.pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entry_गिनती = entries.size();
        bind_group_desc.entries    = entries.तथ्य();
        bind_group_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.CreateBindGroup(&bind_group_desc));
        param_offsets.push_back(param_offset);
    }

    क्रम (माप_प्रकार i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_बफर(ctx->param_arena.बफर, param_offsets[i], dispatches[i].params.तथ्य(),
                                           dispatches[i].params.size() * माप(uint32_t));
    }

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    क्रम (माप_प्रकार i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_गिनती + 2 <= WEBGPU_MAX_PROFILE_QUERY_गिनती);
        स्थिर uint32_t query_begin = ctx->profile_timestamp_query_गिनती++;
        स्थिर uint32_t query_end   = ctx->profile_timestamp_query_गिनती++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.querySet                    = ctx->profile_timestamp_query_set;
        ts_writes.beginning_Of_Pass_Write_सूचकांक   = query_begin;
        ts_writes.end_Of_Pass_Write_सूचकांक         = query_end;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass(&pass_desc);

        pass.SetPipeline(dispatches[i].pipeline.pipeline);
        pass.SetBindGroup(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
        pass.End();
        परिणाम.pipeline_names.push_back(dispatches[i].pipeline.name);
    }
#अन्यथा
    क्रम (माप_प्रकार i = 0; i < dispatches.size(); i++) {
        अगर (ctx->batch_compute_passes) {
            ctx->active_compute_pass.SetPipeline(dispatches[i].pipeline.pipeline);
            ctx->active_compute_pass.SetBindGroup(0, bind_groups[i]);
            ctx->active_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second,
                                                        1);
        } अन्यथा {
            wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass();
            pass.SetPipeline(dispatches[i].pipeline.pipeline);
            pass.SetBindGroup(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
            pass.End();
        }
    }
#पूर्ण_अगर

    वापस परिणाम;
}

अटल webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    वापस ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_group_entries), { wg_x, wg_y } },
    });
}

अटल व्योम ggml_backend_webgpu_बफर_memset(webgpu_global_context & ctx,
                                              wgpu::बफर_2 &          buf,
                                              uint32_t                value,
                                              माप_प्रकार                  offset,
                                              माप_प्रकार                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) offset, (uint32_t) size, value };
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_bind_group_entry(0, buf, 0, buf.Get_आकार()) };
    माप_प्रकार                            bytes_per_wg =
        ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup * ctx->capabilities.memset_bytes_per_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, bytes_per_wg);

    ctx->queue.Write_बफर(ctx->memset_params_buf, 0, params.तथ्य(), params.size() * माप(uint32_t));

    wgpu::BindGroupEntry params_entry = {};
    params_entry.binding              = 1;
    params_entry.बफर               = ctx->memset_params_buf;
    params_entry.offset               = 0;
    params_entry.size                 = WEBGPU_PARAMS_BUF_आकार_BYTES;
    entries.push_back(params_entry);

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
    bind_group_desc.entry_गिनती = entries.size();
    bind_group_desc.entries    = entries.तथ्य();
    bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.SetPipeline(ctx->memset_pipeline.pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.End();

    wgpu::Command_बफर              command  = encoder.Finish();
    std::vector<wgpu::Command_बफर> commands = { command };
    ctx->queue.Submit(commands.size(), commands.तथ्य());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

अटल स्थिर अक्षर * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    वापस ctx->name.c_str();
}

अटल व्योम ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#अगर_घोषित GGML_WEBGPU_CPU_PROFILE
    std::अ_बाहर << "\n[ggml_webgpu cpu profiling summary]\n";
    द्विगुन योग_cpu = 0.0;
    क्रम (स्थिर स्वतः & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        योग_cpu += kv.second;
    }
    std::अ_बाहर << "ggml_webgpu: total cpu time: " << योग_cpu << " ms\n";
    std::अ_बाहर << "ggml_webgpu: cpu breakdown:\n";
    क्रम (स्थिर स्वतः & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        द्विगुन pct = (योग_cpu > 0.0) ? (kv.second / योग_cpu * 100.0) : 0.0;
        std::अ_बाहर << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    अगर (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::अ_बाहर << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    क्रम (स्थिर स्वतः & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        द्विगुन pct = (योग_cpu > 0.0) ? (kv.second / योग_cpu * 100.0) : 0.0;
        std::अ_बाहर << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#पूर्ण_अगर

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    std::अ_बाहर << "\n[ggml_webgpu gpu profiling summary]\n";
    द्विगुन योग_gpu = 0.0;
    क्रम (स्थिर स्वतः & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        योग_gpu += kv.second;
    }
    std::अ_बाहर << "ggml_webgpu: total gpu time (all shaders): " << योग_gpu << " ms\n";
    std::अ_बाहर << "\nggml_webgpu: gpu breakdown:\n";
    क्रम (स्थिर स्वतः & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        द्विगुन pct = (योग_gpu > 0.0) ? (kv.second / योग_gpu * 100.0) : 0.0;
        std::अ_बाहर << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::जड़ << std::रखो_सटीक(2)
                  << pct << "%)\n";
    }
#पूर्ण_अगर

#अगर defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::अ_बाहर << "ggml_webgpu: gpu/cpu ratio: " << (योग_cpu > 0.0 ? योग_gpu / योग_cpu : 0.0) << "\n";
#पूर्ण_अगर

    मिटाओ ctx;
    मिटाओ backend;
}

अटल webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_pipeline(shader_lib_ctx);

    स्वतः *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());
    स्थिर bool inplace   = decisions->inplace;

    स्थिर uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    स्थिर uint32_t dst_type_आकार = (uint32_t) ggml_type_आकार(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (((स्थिर int32_t *) dst->op_params)[3] / dst_type_आकार),

        (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)),

        1u,
        (uint32_t) (((स्थिर int32_t *) dst->op_params)[0] / dst_type_आकार),
        (uint32_t) (((स्थिर int32_t *) dst->op_params)[1] / dst_type_आकार),
        (uint32_t) (((स्थिर int32_t *) dst->op_params)[2] / dst_type_आकार),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          binding_सूचकांक = 0;
    अगर (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        binding_सूचकांक++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_सूचकांक, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_सूचकांक + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    स्थिर uint32_t ne = (uint32_t) ggml_nelements(dst);

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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_आकार;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    स्थिर uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_आकार);
    स्थिर uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    स्थिर int32_t s0 = ggml_get_op_params_i32(dst, 0);
    स्थिर int32_t s1 = ggml_get_op_params_i32(dst, 1);
    स्थिर int32_t p0 = ggml_get_op_params_i32(dst, 2);
    स्थिर int32_t p1 = ggml_get_op_params_i32(dst, 3);
    स्थिर int32_t d0 = ggml_get_op_params_i32(dst, 4);
    स्थिर int32_t d1 = ggml_get_op_params_i32(dst, 5);

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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
अटल webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    स्थिर int32_t s0 = ggml_get_op_params_i32(dst, 0);
    स्थिर int32_t s1 = ggml_get_op_params_i32(dst, 1);
    स्थिर int32_t p0 = ggml_get_op_params_i32(dst, 2);
    स्थिर int32_t p1 = ggml_get_op_params_i32(dst, 3);
    स्थिर int32_t d0 = ggml_get_op_params_i32(dst, 4);
    स्थिर int32_t d1 = ggml_get_op_params_i32(dst, 5);

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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    स्थिर bool      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    स्वतः *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    स्थिर int32_t s0    = ggml_get_op_params_i32(dst, 0);
    स्थिर int32_t s1    = ggml_get_op_params_i32(dst, 1);
    स्थिर int32_t p0    = ggml_get_op_params_i32(dst, 2);
    स्थिर int32_t p1    = ggml_get_op_params_i32(dst, 3);
    स्थिर int32_t d0    = ggml_get_op_params_i32(dst, 4);
    स्थिर int32_t d1    = ggml_get_op_params_i32(dst, 5);
    स्थिर bool    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    स्थिर uint32_t KW = src0->ne[0];
    स्थिर uint32_t KH = is_2D ? src0->ne[1] : 1;
    स्थिर uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    स्थिर uint32_t IW = src1->ne[0];
    स्थिर uint32_t IH = is_2D ? src1->ne[1] : 1;
    स्थिर uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    स्थिर uint32_t OW = dst->ne[1];
    स्थिर uint32_t OH = is_2D ? dst->ne[2] : 1;

    स्थिर uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_आकार(src1->type));
    स्थिर uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)) : 0;
    स्थिर uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type));
    स्थिर uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type));

    स्थिर uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_आकार(dst->type));
    स्थिर uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type));
    स्थिर uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)) : 0;
    स्थिर uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)) :
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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    स्वतः *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.लो());

    स्थिर uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_per_wg);

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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    स्थिर uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_आकार_2);
    स्थिर uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    अगर (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    अगर (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    अगर (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_scan_pipeline(shader_lib_ctx, xbc_overlap);
    स्वतः *          decisions = static_cast<ggml_webgpu_ssm_scan_shader_decisions *>(pipeline.context.लो());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t offset_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    uint32_t offset_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type));
    uint32_t offset_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_आकार(src3->type));
    uint32_t offset_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_आकार(src4->type));
    uint32_t offset_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_आकार(src5->type));
    uint32_t offset_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_आकार(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    अगर (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
    } अन्यथा अगर (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    } अन्यथा अगर (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    माप_प्रकार xbc_bind_offset = 0;
    माप_प्रकार xbc_bind_आकार   = 0;
    अगर (xbc_overlap > 0) {
        xbc_bind_offset = xbc_merged_range.offset;
        xbc_bind_आकार   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        offset_x,
        offset_dt,
        offset_A,
        offset_B,
        offset_C,
        offset_ids,
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

    uint32_t binding_अंक = 0;

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, src0),
    };
    // xbc_merged binding
    अगर (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_अंक++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_offset, xbc_bind_आकार));
    }
    // x
    अगर (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, src1));
    }
    // B
    अगर (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, src4));
    }
    // C
    अगर (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक++, dst));

    स्थिर uint32_t योग_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    स्थिर uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(योग_wg, max_wg_per_dim, wg_x, wg_y);

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    स्थिर uint32_t s_v      = (uint32_t) src2->ne[0];
    स्थिर uint32_t h        = (uint32_t) src2->ne[1];
    स्थिर uint32_t n_tokens = (uint32_t) src2->ne[2];
    स्थिर uint32_t n_seqs   = (uint32_t) src2->ne[3];
    स्थिर uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    स्थिर भग्न    scale    = 1.0f / sqrtf((भग्न) s_v);
    uint32_t       scale_u32;
    स_नकल(&scale_u32, &scale, माप(scale_u32));

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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0), ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2), ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4), ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, dst),
    };

    अगर (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst_fuse));
    }

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

अटल std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    अगर (ggml_is_empty(src) || ggml_is_empty(idx)) {
        वापस std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_rows_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_set_rows_shader_decisions *>(pipeline.context.लो());

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

    std::vector<wgpu::BindGroupEntry> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));

    अगर (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->set_rows_dev_error_buf, 0,
                                                            ctx->set_rows_dev_error_buf.Get_आकार()));
    }

    uint32_t threads;
    अगर (ggml_is_quantized(dst->type)) {
        स्थिर uint32_t blocks_per_row = src->ne[0] / ggml_blck_आकार(dst->type);
        threads =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_blocks ? (blocks_per_row / 2) : blocks_per_row);
    } अन्यथा अगर (decisions->vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } अन्यथा {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(threads, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
अटल std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_आकार_entry(uint32_t wg_आकार) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].कुंजी   = "wg_size";
    constants[0].value = wg_आकार;
    वापस constants;
}

अटल webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    स्थिर bool float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    स्थिर uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type));
    स्थिर uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type));
    स्थिर uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_आकार(src->type));
    स्थिर uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_आकार(src->type));
    स्थिर uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_आकार(src->type));

    स्थिर bool vec4_aligned = offset_src % 4 == 0 && offset_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_rows_pipeline(shader_lib_ctx, vec4_aligned);
    स्वतः *          decisions = static_cast<ggml_webgpu_get_rows_shader_decisions *>(pipeline.context.लो());

    std::vector<uint32_t> params = { offset_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_आकार(idx->type)),
                                     offset_dst,
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

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst) };

    uint32_t blocks_per_row = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t योग_rows     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t योग_threads  = float_parallel ? blocks_per_row * योग_rows : योग_rows;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(योग_threads, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल व्योम ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    स्थिर माप_प्रकार dst_offset           = ggml_webgpu_tensor_offset(dst);
    स्थिर माप_प्रकार q8_src1_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment);
    स्थिर माप_प्रकार q8_src1_binding_आकार = ROUNDUP_POW2(
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

    std::vector<wgpu::BindGroupEntry> q8_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_offset, q8_src1_binding_आकार)
    };

    स्वतः q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.लो());

    uint32_t       q8_wg_आकार     = q8_decisions->wg_आकार;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    स्थिर uint32_t wg_per_vec     = (src0->ne[0] / 4 + (q8_wg_आकार - 1)) / q8_wg_आकार;
    स्थिर uint32_t q8_योग_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_per_vec;
    स्थिर uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(q8_योग_wg, max_wg_per_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

अटल webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    bool use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    bool use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_product,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_आकार              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_आकार        = ctx->global_ctx->capabilities.min_subgroup_आकार;
    shader_lib_ctx.max_subgroup_आकार        = ctx->global_ctx->capabilities.max_subgroup_आकार;
    shader_lib_ctx.supports_dot_product     = ctx->global_ctx->capabilities.supports_dot_product;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    स्थिर bool src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    अगर (use_mat_vec) {
        अगर (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } अन्यथा {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    माप_प्रकार   merged_offset = 0;
    माप_प्रकार   merged_आकार   = 0;
    अगर (src_overlap) {
        स्थिर ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_आकार   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { offset_src0,
                                     offset_src1,
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
    std::vector<wgpu::BindGroupEntry> entries = {};
    अगर (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        स्वतः & mmvq_qq8_entry = dispatches[0].bind_group_entries[1];
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_entry.offset,
                                                            mmvq_qq8_entry.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    } अन्यथा अगर (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    स्थिर uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    अगर (use_mat_vec) {
        स्वतः * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.लो());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
        uint32_t योग_wg      = output_groups * batches;
        compute_2d_workgroups(योग_wg, max_wg_per_dim, wg_x, wg_y);
    } अन्यथा {
        स्वतः * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.लो());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        अगर (decisions->use_subgroup_matrix) {
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
        compute_2d_workgroups(योग_wg, max_wg_per_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    वापस ggml_backend_webgpu_build_multi(ctx, dispatches);
}

अटल webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    स्थिर uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    स्थिर uint32_t param_n_expert_used = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_आकार(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    स्वतः * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.लो());

    स्थिर uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
    uint32_t       योग_wg       = output_groups * param_n_expert_used;
    compute_2d_workgroups(योग_wg, max_wg_per_dim, wg_x, wg_y);

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
अटल uint32_t ggml_webgpu_mul_mat_id_max_tiles(स्थिर ggml_tensor * dst) {
    स्थिर uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    स्थिर uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    वापस CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_आकार_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
अटल माप_प्रकार ggml_webgpu_mul_mat_id_tiles_nbytes(स्थिर ggml_tensor * dst) {
    वापस (1 + 3 * (माप_प्रकार) ggml_webgpu_mul_mat_id_max_tiles(dst)) * माप(uint32_t);
}

अटल webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    अगर (dst->ne[2] == 1) {
        वापस ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    स्थिर uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    स्थिर uint32_t param_n_expert_used = (uint32_t) dst->ne[1];
    स्थिर uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)),
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_आकार(src2->type)),
    };

    स्थिर माप_प्रकार dst_offset       = ggml_webgpu_tensor_offset(dst);
    स्थिर माप_प्रकार ids_cache_nbytes = src2->ne[0] * src1->ne[2] * माप(uint32_t);
    स्थिर माप_प्रकार ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    स्थिर माप_प्रकार ids_cache_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment);
    स्थिर माप_प्रकार ids_tiles_align_offset =
        ROUNDUP_POW2(ids_cache_align_offset + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment);

    स्थिर माप_प्रकार ids_cache_bind_आकार = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    स्थिर माप_प्रकार ids_tiles_bind_आकार = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::BindGroupEntry> gather_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_आकार),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_आकार),
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
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::BindGroupEntry> main_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_आकार(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_आकार(ctx, dst)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_आकार),
        ggml_webgpu_make_bind_group_entry(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_आकार),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    स्वतः * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.लो());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_आकार_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_आकार_N);

    स्थिर uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_आकार_m;
    स्थिर uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    स्थिर uint32_t योग_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    वापस ggml_backend_webgpu_build_multi(ctx, dispatches);
}

काष्ठा ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> entries;
    माप_प्रकार                            kv_bind_offset = 0;
    माप_प्रकार                            kv_bind_आकार   = 0;
    bool                              has_mask       = false;
    bool                              has_sinks      = false;
    bool                              kv_overlap     = false;
};

अटल bool ggml_webgpu_flash_attn_use_vec_path(स्थिर webgpu_global_context & global_ctx,
                                                स्थिर ggml_tensor *           Q,
                                                स्थिर ggml_tensor *           K,
                                                स्थिर ggml_tensor *           V) {
    स्थिर माप_प्रकार storage_offset_alignment = global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment;

    स्थिर bool k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment);
    स्थिर bool v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);

    स्थिर uint32_t k_vec_शीर्ष_align =
        ggml_is_quantized(K->type) ? ggml_blck_आकार(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    स्थिर uint32_t v_vec_शीर्ष_align =
        ggml_is_quantized(V->type) ? ggml_blck_आकार(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    स्थिर bool kv_vec_शीर्ष_dims_aligned = Q->ne[0] % k_vec_शीर्ष_align == 0 && V->ne[0] % v_vec_शीर्ष_align == 0;

    वापस global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_शीर्ष_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

अटल ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    mask,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    भग्न scale         = ggml_get_op_params_f32(dst, 0);
    भग्न max_bias      = ggml_get_op_params_f32(dst, 1);
    भग्न logit_softcap = ggml_get_op_params_f32(dst, 2);
    अगर (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    भग्न n_शीर्ष_log2 = भग्न(1u << (uint32_t) न्यूनमान(log2(Q->ne[2])));
    भग्न m0          = powf(2.0f, -(max_bias) / n_शीर्ष_log2);
    भग्न m1          = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = mask;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    op.shader_lib_ctx.max_wg_आकार              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    op.shader_lib_ctx.wg_mem_limit_bytes       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_आकार;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_आकार        = ctx->global_ctx->capabilities.min_subgroup_आकार;
    op.shader_lib_ctx.max_subgroup_आकार        = ctx->global_ctx->capabilities.max_subgroup_आकार;

    op.has_mask   = mask != nullptr;
    op.has_sinks  = sinks != nullptr;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t offset_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_आकार(K->type));
    uint32_t offset_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_आकार(V->type));
    अगर (op.kv_overlap) {
        स्थिर ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_offset                                   = merged_range.offset;
        op.kv_bind_आकार                                     = merged_range.size;
        offset_k                                            = ggml_webgpu_tensor_merged_element_offset(K, merged_range);
        offset_v                                            = ggml_webgpu_tensor_merged_element_offset(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_आकार(Q->type)),
        offset_k,
        offset_v,
        op.has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_आकार(mask->type)) : 0,
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
        op.has_mask ? (uint32_t) (mask->nb[3] / ggml_type_आकार(mask->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_शीर्ष_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
    };
    अगर (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_आकार));
    } अन्यथा {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
    }
    uint32_t binding_सूचकांक = op.kv_overlap ? 2u : 3u;
    अगर (op.has_mask) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_सूचकांक++, mask));
    }
    अगर (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_सूचकांक++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_सूचकांक++, dst));

    वापस op;
}

अटल uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    स्थिर uint64_t kv_span = (uint64_t) kv_tile;
    जबतक ((2u * nwg * kv_span) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    वापस std::min(nwg, vec_nwg_cap);
}

अटल webgpu_encoded_op ggml_webgpu_flash_attn_direct(webgpu_context & ctx, स्थिर ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    स्वतः *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.लो());
    uint32_t        wg_per_शीर्ष = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_per_शीर्ष * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    वापस ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

अटल webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             mask,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    स्वतः *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.लो());

    wgpu::बफर_2 blk_buf         = {};
    uint64_t     blk_आकार_bytes  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_गिनती = 0;

    स्थिर uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_आकार;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    स्थिर uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    स्थिर bool     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_base  = 0;
    uint64_t     tmp_आकार_bytes  = 0;
    wgpu::बफर_2 tmp_buf         = {};
    uint64_t     tmp_bind_offset = 0;
    uint64_t     tmp_bind_आकार   = 0;
    स्थिर माप_प्रकार align_bytes     = ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment;
    स्थिर माप_प्रकार dst_offset      = ggml_webgpu_tensor_offset(dst);
    माप_प्रकार       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);

    अगर (use_vec_reduce) {
        स्थिर uint64_t tmp_तथ्य_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        स्थिर uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_base                 = tmp_तथ्य_elems;
        tmp_आकार_bytes =
            ROUNDUP_POW2((tmp_तथ्य_elems + tmp_stats_elems) * माप(भग्न), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_आकार   = tmp_आकार_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_आकार_bytes, align_bytes);
    } अन्यथा {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_आकार_bytes  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_आकार   = tmp_आकार_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_आकार_bytes, align_bytes);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::BindGroupEntry> blk_entries;
    अगर (op.has_mask) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        स्थिर uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_आकार(mask->type));
        blk_batch_गिनती             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        स्थिर uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_गिनती;
        blk_आकार_bytes              = ROUNDUP_POW2(blk_elems * माप(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        स्थिर ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_आकार(mask->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(mask),
                                              ggml_webgpu_tensor_align_offset(ctx, mask),
                                              ggml_webgpu_tensor_binding_आकार(ctx, mask)),
            ggml_webgpu_make_bind_group_entry(1, blk_buf, scratch_offset, blk_आकार_bytes),
        };
        scratch_offset = ROUNDUP_POW2(scratch_offset + blk_आकार_bytes, align_bytes);
    }

    std::vector<uint32_t> split_params = op.params;
    अगर (op.has_mask) {
        split_params.push_back(0u);                     // blk_base
        split_params.push_back(blk_nblk0);              // blk_nblk0
        split_params.push_back(blk_nblk1);              // blk_nblk1
    }
    split_params.push_back(0u);                         // tmp_data_base
    split_params.push_back((uint32_t) tmp_stats_base);  // tmp_stats_base
    split_params.push_back(nwg);                        // nwg

    std::vector<wgpu::BindGroupEntry> split_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_offset(ctx, Q),
                                          ggml_webgpu_tensor_binding_आकार(ctx, Q)),
    };
    अगर (op.kv_overlap) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_आकार));
    } अन्यथा {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, K)));
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, V)));
    }
    uint32_t split_binding_सूचकांक = op.kv_overlap ? 2u : 3u;
    अगर (op.has_mask) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_सूचकांक++, ggml_webgpu_tensor_buf(mask),
                                                                  ggml_webgpu_tensor_align_offset(ctx, mask),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, mask)));
    }
    अगर (op.has_sinks) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_सूचकांक++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_offset(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_आकार(ctx, sinks)));
    }
    अगर (op.has_mask) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(split_binding_सूचकांक++, blk_buf, blk_entries[1].offset, blk_आकार_bytes));
    }
    split_entries.push_back(
        ggml_webgpu_make_bind_group_entry(split_binding_सूचकांक++, tmp_buf, tmp_bind_offset, tmp_bind_आकार));
    split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_सूचकांक++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_offset(ctx, dst),
                                                              ggml_webgpu_tensor_binding_आकार(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::BindGroupEntry> reduce_entries;
    अगर (use_vec_reduce) {
        स्थिर uint32_t reduce_sg_आकार = ctx->global_ctx->capabilities.max_subgroup_आकार;
        स्थिर uint32_t reduce_wg_आकार = std::max(
            reduce_sg_आकार,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_आकार,
                                          ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup));
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
            (uint32_t) tmp_stats_base,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_group_entry(0, tmp_buf, tmp_bind_offset, tmp_आकार_bytes),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                              ggml_webgpu_tensor_binding_आकार(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    स्थिर uint64_t split_wg_योग = (uint64_t) wg_x * nwg;
    GGML_ASSERT(split_wg_योग <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    अगर (op.has_mask) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_गिनती }
        });
    }
    dispatches.push_back({
        pipeline, std::move(split_params), std::move(split_entries), { (uint32_t) split_wg_योग, 1u }
    });
    अगर (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    वापस ggml_backend_webgpu_build_multi(ctx, dispatches);
}

अटल webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    mask,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, mask, sinks, dst);
    अगर (ggml_webgpu_flash_attn_use_vec_path(ctx->global_ctx, Q, K, V)) {
        वापस ggml_webgpu_flash_attn_vec(ctx, Q, K, V, mask, sinks, dst, std::move(op));
    }
    वापस ggml_webgpu_flash_attn_direct(ctx, op);
}

अटल webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    स्वतः *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());
    स्थिर bool inplace   = decisions->inplace;

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
    अगर (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        चयन (unary_op) {
            हाल GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    भग्न alpha_n = ggml_get_op_params_f32(dst, 1);
                    भग्न alpha_p = ggml_get_op_params_f32(dst, 2);
                    भग्न beta    = ggml_get_op_params_f32(dst, 3);
                    भग्न eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    अवरोध;
                }
            शेष:
                अवरोध;
        }
    } अन्यथा अगर (dst->op == GGML_OP_CLAMP) {
        भग्न clamp_min = ggml_get_op_params_f32(dst, 0);
        भग्न clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } अन्यथा अगर (dst->op == GGML_OP_FILL) {
        भग्न fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, effective_src),
    };
    अगर (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_द्विआधारी_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    स्थिर bool      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_द्विआधारी_pipeline(shader_lib_ctx, src_overlap);
    स्वतः *          decisions   = static_cast<ggml_webgpu_द्विआधारी_shader_decisions *>(pipeline.context.लो());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    माप_प्रकार src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
    माप_प्रकार src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    माप_प्रकार   merged_offset = 0;
    माप_प्रकार   merged_आकार   = 0;
    अगर (decisions->src_overlap) {
        स्थिर ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_आकार   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        offset_src0,
        offset_src1,
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

    std::vector<wgpu::BindGroupEntry> entries;

    अगर (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_आकार(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_आकार(ctx, src1)));
        अगर (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_add_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_add_id_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

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

    std::vector<wgpu::BindGroupEntry> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));

    अगर (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       योग_wg       = ggml_nrows(dst);
    स्थिर uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(योग_wg, max_wg_per_dim, wg_x, wg_y);

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    अगर (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        वापस {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    स्थिर bool      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    स्वतः *          decisions   = static_cast<ggml_webgpu_द्विआधारी_shader_decisions *>(pipeline.context.लो());

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
    माप_प्रकार   merged_offset = 0;
    माप_प्रकार   merged_आकार   = 0;
    अगर (decisions->src_overlap) {
        अगर (ggml_nbytes(src0) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src1);
            merged_आकार   = ggml_webgpu_tensor_binding_आकार(ctx, src1);
            offset_src0   = 0;
            offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type));
        } अन्यथा अगर (ggml_nbytes(src1) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
            merged_आकार   = ggml_webgpu_tensor_binding_आकार(ctx, src0);
            offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
            offset_src1   = 0;
        } अन्यथा {
            स्थिर ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_offset = merged_range.offset;
            merged_आकार   = merged_range.size;
            offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
            offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     offset_src0,
                                     offset_src1,
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

    std::vector<wgpu::BindGroupEntry> entries = {};
    अगर (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
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

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    स्वतः *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ne, decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    अगर (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } अन्यथा अगर (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } अन्यथा {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t offset_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_आकार(rn_src->type));
    uint32_t offset_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_आकार(mul_src->type));
    माप_प्रकार merged_offset = 0;
    माप_प्रकार merged_आकार   = 0;

    std::vector<uint32_t> params = {
        offset_rn_src,
        offset_mul_src,
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

    std::vector<wgpu::BindGroupEntry> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    स्थिर bool      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    स्वतः *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.लो());

    अगर (decisions->src_overlap) {
        स्थिर ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_offset  = merged_range.offset;
        merged_आकार    = merged_range.size;
        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
        params[0]      = offset_rn_src;
        params[1]      = offset_mul_src;
    }

    अगर (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
    } अन्यथा अगर (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_आकार));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
    स्वतः *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
    अगर (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    स्वतः * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    स्थिर bool inplace         = decisions->inplace;
    स्थिर पूर्णांक  has_freq_factor = (src2 != nullptr);

    स्थिर पूर्णांक n_dims     = ((int32_t *) dst->op_params)[1];
    स्थिर पूर्णांक mode       = ((int32_t *) dst->op_params)[2];
    स्थिर पूर्णांक n_ctx_orig = ((int32_t *) dst->op_params)[4];
    स्थिर पूर्णांक n_offs     = ((int32_t *) dst->op_params)[15];

    भग्न freq_base;
    भग्न freq_scale;
    भग्न ext_factor;
    भग्न attn_factor;
    भग्न beta_fast;
    भग्न beta_slow;
    स_नकल(&freq_base, (int32_t *) dst->op_params + 5, माप(भग्न));
    स_नकल(&freq_scale, (int32_t *) dst->op_params + 6, माप(भग्न));
    स_नकल(&ext_factor, (int32_t *) dst->op_params + 7, माप(भग्न));
    स_नकल(&attn_factor, (int32_t *) dst->op_params + 8, माप(भग्न));
    स_नकल(&beta_fast, (int32_t *) dst->op_params + 9, माप(भग्न));
    स_नकल(&beta_slow, (int32_t *) dst->op_params + 10, माप(भग्न));

    पूर्णांक sections[4];
    स_नकल(sections, (int32_t *) dst->op_params + 11, 4 * माप(पूर्णांक));

    भग्न theta_scale = powf(freq_base, -2.0f / n_dims);

    भग्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)),
        src2 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)) : 0,
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
        ggml_webgpu_u32_from_f32(attn_factor),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_factor),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    अगर (has_freq_factor) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
    }
    अगर (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    स्थिर bool      src_overlap = src1 != nullptr && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    स्वतः * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.लो());

    स्थिर पूर्णांक split = (src1 != nullptr);

    uint32_t offset_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type));
    uint32_t offset_src1 =
        src1 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)) : 0;
    माप_प्रकार merged_offset = 0;
    माप_प्रकार merged_आकार   = 0;
    अगर (decisions->src_overlap) {
        स्थिर ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_आकार   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)) :
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
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  // limit
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          dst_binding = 1;
    अगर (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_आकार));
    } अन्यथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    }
    अगर (split && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    स्वतः *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

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
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };

    अगर (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    स्वतः *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    स्थिर bool inplace     = decisions->inplace;
    स्थिर पूर्णांक  has_mask    = (src1 != nullptr);
    स्थिर पूर्णांक  has_sink    = (src2 != nullptr);
    भग्न      max_bias    = ggml_get_op_params_f32(dst, 1);
    भग्न      n_शीर्ष_log2 = भग्न(1u << (uint32_t) न्यूनमान(log2(src0->ne[2])));
    भग्न      m0          = powf(2.0f, -(max_bias) / n_शीर्ष_log2);
    भग्न      m1          = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_आकार(src0->type)),
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_आकार(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_आकार(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_आकार(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_आकार(src0->type)),
        has_mask ? (uint32_t) (src1->nb[1] / ggml_type_आकार(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[2] / ggml_type_आकार(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[3] / ggml_type_आकार(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_आकार(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_आकार(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_mask ? (uint32_t) src1->ne[2] : 0,
        has_mask ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_शीर्ष_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_bind_group_entry(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
        ggml_webgpu_tensor_binding_आकार(ctx, src0)) };
    uint32_t                          binding_अंक = 1;
    अगर (has_mask) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_अंक, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_offset(ctx, src1),
                                                            ggml_webgpu_tensor_binding_आकार(ctx, src1)));
        binding_अंक++;
    }
    अगर (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक, src2));
        binding_अंक++;
    }
    अगर (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अंक, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = ggml_nrows(dst);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

अटल webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool is_शीर्ष_k = dst->op == GGML_OP_शीर्ष_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_आकार;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    स्वतः * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.लो());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    स्थिर uint32_t src_ne0 = (uint32_t) src->ne[0];
    स्थिर uint32_t nrows   = (uint32_t) ggml_nrows(src);
    स्थिर uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_आकार);
    स्थिर uint32_t block_आकार_2 =
        is_शीर्ष_k ? std::min(argsort_decisions->wg_आकार, (uint32_t) dst->ne[0]) : argsort_decisions->wg_आकार;
    uint32_t out_ne0 = src_ne0;
    अगर (is_शीर्ष_k) {
        अगर (npr > 1) {
            स्थिर uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_आकार;
            out_ne0                  = (npr - 1) * block_आकार_2 + std::min(last_tile, block_आकार_2);
        } अन्यथा {
            out_ne0 = block_आकार_2;
        }
    }

    uint32_t merge_len    = block_आकार_2;
    uint32_t merge_passes = 0;
    जबतक (merge_len < out_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    स्थिर bool start_in_tmp = (merge_passes % 2) == 1;

    स्थिर माप_प्रकार dst_offset = ggml_webgpu_tensor_offset(dst);
    स्थिर माप_प्रकार idx_nbytes = out_ne0 * ggml_nrows(dst) * माप(int32_t);
    स्थिर माप_प्रकार tmp_offset =
        ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment);
    स्थिर माप_प्रकार tmp_binding_आकार = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    स्थिर माप_प्रकार dst_binding_आकार =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    स्थिर uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type));
    स्थिर uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type));
    स्थिर uint32_t offset_tmp  = 0;
    स्थिर uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_आकार(src->type));
    स्थिर uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_आकार(src->type));
    स्थिर uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_आकार(src->type));
    स्थिर uint32_t stride_idx1 = out_ne0;
    स्थिर uint32_t stride_idx2 = out_ne0 * (uint32_t) dst->ne[1];
    स्थिर uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    स्थिर uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    स्थिर माप_प्रकार   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    स्थिर माप_प्रकार   init_binding_आकार = start_in_tmp ? tmp_binding_आकार : dst_binding_आकार;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_आकार_2,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    स्थिर uint32_t योग_wg_init  = npr * nrows;
    स्थिर uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(योग_wg_init, max_wg_per_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::BindGroupEntry> init_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), init_align_offset, init_binding_आकार)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    अगर (merge_passes == 0) {
        वापस ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    bool     in_is_tmp = start_in_tmp;
    uint32_t len       = block_आकार_2;
    जबतक (len < out_ne0) {
        स्थिर uint32_t nm = CEIL_DIV(out_ne0, 2 * len);

        स्थिर bool     out_is_tmp  = !in_is_tmp;
        स्थिर uint32_t offset_in   = in_is_tmp ? offset_tmp : offset_dst;
        स्थिर uint32_t offset_out  = out_is_tmp ? offset_tmp : offset_dst;
        स्थिर माप_प्रकार   align_in    = in_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        स्थिर माप_प्रकार   align_out   = out_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        स्थिर माप_प्रकार   आकार_in     = in_is_tmp ? tmp_binding_आकार : dst_binding_आकार;
        स्थिर माप_प्रकार   आकार_out    = out_is_tmp ? tmp_binding_आकार : dst_binding_आकार;
        स्थिर uint32_t शीर्ष_k_out   = (is_शीर्ष_k && nm == 1) ? (uint32_t) dst->ne[0] : out_ne0;
        स्थिर uint32_t stride_out1 = शीर्ष_k_out;
        स्थिर uint32_t stride_out2 = शीर्ष_k_out * (uint32_t) dst->ne[1];
        स्थिर uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> merge_params = { offset_src,
                                               offset_in,
                                               offset_out,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               out_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               शीर्ष_k_out,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::BindGroupEntry> merge_entries = {
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), align_in, आकार_in),
            ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), align_out, आकार_out)
        };

        uint32_t       wg_x_merge;
        uint32_t       wg_y_merge;
        स्थिर uint32_t योग_wg_merge = nm * nrows;
        compute_2d_workgroups(योग_wg_merge, max_wg_per_dim, wg_x_merge, wg_y_merge);

        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    वापस ggml_backend_webgpu_build_multi(ctx, dispatches);
}

अटल webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        योग_wg = ggml_nrows(dst);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल webgpu_encoded_op ggml_webgpu_योग_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    bool                  योग_योग = dst->op == GGML_OP_योग;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_आकार(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_आकार(dst->type)),
                                     योग_योग ? 0 : (uint32_t) (src->nb[1] / ggml_type_आकार(src->type)),
                                     योग_योग ? 0 : (uint32_t) (src->nb[2] / ggml_type_आकार(src->type)),
                                     योग_योग ? 0 : (uint32_t) (src->nb[3] / ggml_type_आकार(src->type)),
                                     योग_योग ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     योग_योग ? 1 : (uint32_t) src->ne[1],
                                     योग_योग ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_योग_rows_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = योग_योग ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

अटल bool ggml_webgpu_can_fuse_rms_norm_mul(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    अगर (!ggml_can_fuse(cgraph, गाँठ_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        वापस false;
    }

    // additional constraints specific to this fusion
    स्थिर ggml_tensor * rms_norm = cgraph->nodes[गाँठ_idx];
    स्थिर ggml_tensor * mul      = cgraph->nodes[गाँठ_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    अगर (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        वापस false;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    अगर (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        वापस false;
    }
    // rms_norm shader assumes contiguous rows
    अगर (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        वापस false;
    }

    वापस true;
}

अटल bool ggml_webgpu_can_fuse_gdn_cache(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, पूर्णांक & अंक_encoded_ops) {
    स्थिर ggml_tensor * gdn = cgraph->nodes[गाँठ_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    अगर (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        वापस false;
    }

    स्थिर ggml_tensor * src_v     = gdn->src[2];
    स्थिर int64_t       S_v       = src_v->ne[0];
    स्थिर int64_t       H         = src_v->ne[1];
    स्थिर int64_t       n_tokens  = src_v->ne[2];
    स्थिर int64_t       n_seqs    = src_v->ne[3];
    स्थिर int64_t       D         = S_v * S_v * H;
    स्थिर int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    स्थिर int64_t       n_written = std::min<int64_t>(n_tokens, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    स्थिर माप_प्रकार पुच्छ_off = ggml_row_आकार(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    स्थिर ggml_tensor * cpy     = nullptr;
    पूर्णांक                 cpy_idx = 0;
    क्रम (पूर्णांक j = गाँठ_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        स्थिर ggml_tensor * n = cgraph->nodes[j];
        अगर (ggml_op_is_empty(n->op) || ggml_is_empty(n)) {
            जारी;
        }
        अगर (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            वापस false;
        }
        cpy     = n;
        cpy_idx = j;
    }
    अगर (cpy == nullptr) {
        वापस false;
    }

    स्थिर ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    स्थिर ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    अगर (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != पुच्छ_off ||
        !ggml_is_contiguous(cpy_src)) {
        वापस false;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    स्थिर std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    अगर (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->तथ्य == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_आकार(GGML_TYPE_F32) || cpy_dst->nb[1] != (माप_प्रकार) ggml_row_आकार(GGML_TYPE_F32, D)) {
        वापस false;
    }

    अंक_encoded_ops = cpy_idx - गाँठ_idx + 1;

    वापस true;
}

अटल webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    स्थिर uint32_t        mode_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
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

                                     mode_flags };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_आकार = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    स्वतः *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.लो());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t योग_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_आकार);
    compute_2d_workgroups(योग_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    वापस ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
अटल std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           पूर्णांक            गाँठ_idx,
                                                           पूर्णांक &          अंक_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  गाँठ  = nodes[गाँठ_idx];

    अगर (ggml_is_empty(गाँठ)) {
        वापस std::nullopt;
    }
    अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        वापस std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << गाँठ << ", " << ggml_op_name(गाँठ->op) << ")");

    ggml_tensor * src0 = गाँठ->src[0];
    ggml_tensor * src1 = गाँठ->src[1];
    ggml_tensor * src2 = गाँठ->src[2];

    चयन (गाँठ->op) {
            // no-ops
        हाल GGML_OP_NONE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_RESHAPE:
            वापस std::nullopt;
        हाल GGML_OP_DUP:
        हाल GGML_OP_CPY:
        हाल GGML_OP_CONT:
            वापस ggml_webgpu_cpy(ctx, src0, गाँठ);
        हाल GGML_OP_SET:
            वापस ggml_webgpu_set(ctx, src0, src1, गाँठ);
        हाल GGML_OP_SET_ROWS:
            वापस ggml_webgpu_set_rows(ctx, src0, src1, गाँठ);
        हाल GGML_OP_GET_ROWS:
            वापस ggml_webgpu_get_rows(ctx, src0, src1, गाँठ);
        हाल GGML_OP_MUL_MAT:
            वापस ggml_webgpu_mul_mat(ctx, src0, src1, गाँठ);
        हाल GGML_OP_MUL_MAT_ID:
            वापस ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, गाँठ);
        हाल GGML_OP_FLASH_ATTN_EXT:
            वापस ggml_webgpu_flash_attn(ctx, src0, src1, src2, गाँठ->src[3], गाँठ->src[4], गाँठ);
        हाल GGML_OP_ADD:
        हाल GGML_OP_SUB:
        हाल GGML_OP_MUL:
        हाल GGML_OP_DIV:
            वापस ggml_webgpu_द्विआधारी_op(ctx, src0, src1, गाँठ);
        हाल GGML_OP_ADD_ID:
            वापस ggml_webgpu_add_id(ctx, src0, src1, src2, गाँठ);
        हाल GGML_OP_CONCAT:
            वापस ggml_webgpu_concat(ctx, src0, src1, गाँठ);
        हाल GGML_OP_REPEAT:
            वापस ggml_webgpu_repeat(ctx, src0, गाँठ);
        हाल GGML_OP_RMS_NORM:
            अगर (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, गाँठ_idx)) {
                अंक_encoded_ops        = 2;
                ggml_tensor * mul_गाँठ = nodes[गाँठ_idx + 1];
                वापस ggml_webgpu_rms_norm_mul(ctx, src0, गाँठ, mul_गाँठ->src[0], mul_गाँठ->src[1], mul_गाँठ);
            } अन्यथा {
                वापस ggml_webgpu_row_norm(ctx, src0, गाँठ);
            }
        हाल GGML_OP_NORM:
        हाल GGML_OP_L2_NORM:
            वापस ggml_webgpu_row_norm(ctx, src0, गाँठ);
        हाल GGML_OP_ROPE:
            वापस ggml_webgpu_rope(ctx, src0, src1, src2, गाँठ);
        हाल GGML_OP_GLU:
            वापस ggml_webgpu_glu(ctx, src0, src1, गाँठ);
        हाल GGML_OP_SCALE:
            वापस ggml_webgpu_scale(ctx, src0, गाँठ);
        हाल GGML_OP_SOFT_MAX:
            वापस ggml_webgpu_soft_max(ctx, src0, src1, src2, गाँठ);
        हाल GGML_OP_UNARY:
        हाल GGML_OP_CLAMP:
        हाल GGML_OP_FILL:
        हाल GGML_OP_LOG:
        हाल GGML_OP_SQR:
        हाल GGML_OP_SQRT:
        हाल GGML_OP_SIN:
        हाल GGML_OP_COS:
        हाल GGML_OP_DIAG:
        हाल GGML_OP_TRI:
            वापस ggml_webgpu_unary_op(ctx, src0, गाँठ);
        हाल GGML_OP_SOLVE_TRI:
            वापस ggml_webgpu_solve_tri(ctx, src0, src1, गाँठ);
        हाल GGML_OP_SSM_CONV:
            वापस ggml_webgpu_ssm_conv(ctx, src0, src1, गाँठ);
        हाल GGML_OP_SSM_SCAN:
            वापस ggml_webgpu_ssm_scan(ctx, src0, src1, src2, गाँठ->src[3], गाँठ->src[4], गाँठ->src[5], गाँठ->src[6],
                                        गाँठ);
        हाल GGML_OP_GATED_DELTA_NET:
            अगर (ggml_webgpu_can_fuse_gdn_cache(cgraph, गाँठ_idx, अंक_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[गाँठ_idx + अंक_encoded_ops - 1]->src[1];
                वापस ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, गाँठ->src[3], गाँठ->src[4], गाँठ->src[5],
                                                   गाँठ, dst_fuse);
            } अन्यथा {
                वापस ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, गाँठ->src[3], गाँठ->src[4], गाँठ->src[5],
                                                   गाँठ, nullptr);
            }
        हाल GGML_OP_PAD:
            वापस ggml_webgpu_pad(ctx, src0, गाँठ);
        हाल GGML_OP_ARGMAX:
            वापस ggml_webgpu_argmax(ctx, src0, गाँठ);
        हाल GGML_OP_ARGSORT:
        हाल GGML_OP_शीर्ष_K:
            // we reuse the same argsort implementation for top_k
            वापस ggml_webgpu_argsort(ctx, src0, गाँठ);
        हाल GGML_OP_CUMSUM:
            वापस ggml_webgpu_cumsum(ctx, src0, गाँठ);
        हाल GGML_OP_योग:
        हाल GGML_OP_योग_ROWS:
            वापस ggml_webgpu_योग_rows(ctx, src0, गाँठ);
        हाल GGML_OP_CONV_2D:
            वापस ggml_webgpu_conv_2d(ctx, src0, src1, गाँठ);
        हाल GGML_OP_CONV_2D_DW:
            वापस ggml_webgpu_conv_2d_dw(ctx, src0, src1, गाँठ);
        हाल GGML_OP_IM2COL:
            वापस ggml_webgpu_im2col(ctx, src0, src1, गाँठ);
        हाल GGML_OP_UPSCALE:
            वापस ggml_webgpu_upscale(ctx, src0, गाँठ);
        शेष:
            वापस std::nullopt;
    }
}

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
अटल व्योम ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        स्थिर std::vector<std::string> & pipeline_names,
                                                        uint32_t &                       अंक_inflight_batches) {
    अगर (pipeline_names.empty()) {
        वापस;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.ResolveQuerySet(ctx->profile_timestamp_query_set, 0, ctx->profile_timestamp_query_गिनती,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_बफर_To_बफर(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_गिनती * माप(uint64_t));

    wgpu::Command_बफर profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, अंक_inflight_batches);

    स्थिर माप_प्रकार mapped_आकार = ctx->profile_timestamp_query_गिनती * माप(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_गिनती == 2 * pipeline_names.size());

    ggml_backend_webgpu_map_बफर(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,
                                   mapped_आकार);
    स्थिर uint64_t * ts_तथ्य = (स्थिर uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_आकार);

    क्रम (माप_प्रकार i = 0; i < pipeline_names.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        स्थिर द्विगुन elapsed_ms = द्विगुन(ts_तथ्य[2 * i + 1] - ts_तथ्य[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#पूर्ण_अगर

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
अटल व्योम ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & अंक_inflight_batches) {
#अगर_घोषित GGML_WEBGPU_CHECK_SET_ROWS
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_बफर_To_बफर(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                               ctx->set_rows_host_error_buf.Get_आकार());
    wgpu::Command_बफर commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, अंक_inflight_batches);
    ggml_backend_webgpu_map_बफर(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                   ctx->set_rows_host_error_buf.Get_आकार());
    स्थिर uint32_t * error_तथ्य = (स्थिर uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
    अगर (*error_तथ्य) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->set_rows_host_error_buf.Unmap();
#अन्यथा
    GGML_UNUSED(ctx);
    GGML_UNUSED(अंक_inflight_batches);
#पूर्ण_अगर
}

अटल ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_योग_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t अंक_batched_kernels  = 0;
    uint32_t अंक_inflight_batches = 0;
    bool     contains_set_rows    = false;
    पूर्णांक      अंक_encoded_ops      = 1;
    पूर्णांक      गाँठ_idx             = 0;

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_गिनती = 0;
    std::vector<std::string> profile_pipeline_names;
#पूर्ण_अगर

    ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    अगर (ctx->batch_compute_passes) {
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
    }

    जबतक (गाँठ_idx < cgraph->n_nodes) {
        अगर (cgraph->nodes[गाँठ_idx]->op == GGML_OP_SET_ROWS) {
            contains_set_rows = true;
        }
        अगर (स्वतः cmd = ggml_webgpu_encode(ctx, cgraph, गाँठ_idx, अंक_encoded_ops)) {
            commands.push_back(*cmd);
            अंक_batched_kernels += cmd.value().अंक_kernels;
#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_names.insert(profile_pipeline_names.end(), cmd->pipeline_names.begin(),
                                          cmd->pipeline_names.end());
#पूर्ण_अगर
        }

        अगर (अंक_batched_kernels >= ctx->global_ctx->command_submit_batch_आकार) {
            अगर (ctx->active_compute_pass) {
                ctx->active_compute_pass.End();
            }
            अंक_batched_kernels                = 0;
            wgpu::Command_बफर batch_commands = ctx->active_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, अंक_inflight_batches);

            // reset state for next batch
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            अगर (ctx->batch_compute_passes) {
                ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.सफा();
#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            अगर (ctx->profile_timestamp_query_गिनती + 2 * ctx->global_ctx->command_submit_batch_आकार >=
                WEBGPU_MAX_PROFILE_QUERY_गिनती) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, अंक_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_गिनती = 0;
                profile_pipeline_names.सफा();
            }
#पूर्ण_अगर
        }

        गाँठ_idx += अंक_encoded_ops;
        अंक_encoded_ops = 1;
    }

    अगर (ctx->active_compute_pass) {
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
    }

    अगर (अंक_batched_kernels > 0) {
        wgpu::Command_बफर batch_commands = ctx->active_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, अंक_inflight_batches);
        ctx->param_arena.reset();
        commands.सफा();
    }
    ctx->active_command_encoder = nullptr;

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, अंक_inflight_batches);
#पूर्ण_अगर

    अगर (contains_set_rows) {
        ggml_backend_webgpu_check_set_rows(ctx, अंक_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_योग_END(graph_compute, ctx->global_ctx);
    वापस GGML_STATUS_SUCCESS;
}

काष्ठा ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    bool                  recorded = false;
};

अटल ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    स्वतः * event_ctx      = नया ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    स्वतः * event   = नया ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    वापस event;
}

अटल व्योम ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    मिटाओ static_cast<ggml_backend_webgpu_event_context *>(event->context);
    मिटाओ event;
}

अटल व्योम ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    अगर (!event_ctx->recorded) {
        वापस;
    }
    wgpu::WaitStatus status =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    अगर (status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = false;
}

अटल व्योम ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    event_ctx->recorded = true;
}

अटल व्योम ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
}

अटल व्योम ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 स्थिर व्योम *   तथ्य,
                                                 माप_प्रकार         offset,
                                                 माप_प्रकार         size) {
    GGML_UNUSED(backend);
    स्वतः * buf_ctx      = (ggml_backend_webgpu_बफर_context *) tensor->बफर->context;
    माप_प्रकार योग_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_बफर(buf_ctx->बफर, योग_offset, तथ्य, (size / 4) * 4);

    अगर (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        माप_प्रकार remaining_आकार = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        क्रम (माप_प्रकार i = 0; i < remaining_आकार; i++) {
            ((uint8_t *) &val32)[i] = ((स्थिर uint8_t *) तथ्य)[size - remaining_आकार + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, val32,
                                          योग_offset + (size - remaining_आकार), remaining_आकार);
    }
}

अटल व्योम ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

अटल ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
    /* .get_tensor_async        = */ शून्य,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ शून्य,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_record,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ शून्य,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

अटल व्योम ggml_backend_webgpu_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_webgpu_बफर_context * ctx = static_cast<ggml_backend_webgpu_बफर_context *>(बफर->context);
    अगर (ctx != nullptr && ctx->बफर != nullptr) {
        ctx->बफर.Destroy();
        मिटाओ ctx;
    }
}

// Returns the "fake" base pointer.
अटल व्योम * ggml_backend_webgpu_बफर_get_base(ggml_backend_बफर_t बफर) {
    GGML_UNUSED(बफर);
    वापस webgpu_ptr_base;
}

अटल व्योम ggml_backend_webgpu_बफर_memset_tensor(ggml_backend_बफर_t बफर,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     माप_प्रकार                offset,
                                                     माप_प्रकार                size) {
    अगर (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        वापस;
    }

    WEBGPU_CPU_PROFILE_योग_START(memset_tensor);

    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << offset << ", " << size << ")");

    माप_प्रकार योग_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, val32, योग_offset, size);
    WEBGPU_CPU_PROFILE_योग_END(memset_tensor, buf_ctx->global_ctx);
}

अटल व्योम ggml_backend_webgpu_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                                  ggml_tensor *         tensor,
                                                  स्थिर व्योम *          तथ्य,
                                                  माप_प्रकार                offset,
                                                  माप_प्रकार                size) {
    WEBGPU_CPU_PROFILE_योग_START(set_tensor);
    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << तथ्य
                                                              << ", " << offset << ", " << size << ")");

    माप_प्रकार योग_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    अगर (योग_offset % 4 != 0) {
        माप_प्रकार lane = योग_offset % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        माप_प्रकार शीर्ष = std::min<माप_प्रकार>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t शीर्ष_val = 0;
        क्रम (माप_प्रकार i = 0; i < शीर्ष; i++) {
            ((uint8_t *) &शीर्ष_val)[lane + i] = ((स्थिर uint8_t *) तथ्य)[i];
        }
        ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, शीर्ष_val, योग_offset, शीर्ष);

        योग_offset += शीर्ष;
        size -= शीर्ष;
        तथ्य = (स्थिर uint8_t *) तथ्य + शीर्ष;
    }

    अगर (size > 0) {
        buf_ctx->global_ctx->queue.Write_बफर(buf_ctx->बफर, योग_offset, तथ्य, (size / 4) * 4);
    }

    अगर (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        माप_प्रकार remaining_आकार = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        क्रम (माप_प्रकार i = 0; i < remaining_आकार; i++) {
            ((uint8_t *) &val32)[i] = ((स्थिर uint8_t *) तथ्य)[size - remaining_आकार + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, val32,
                                          योग_offset + (size - remaining_आकार), remaining_आकार);
    }
    WEBGPU_CPU_PROFILE_योग_END(set_tensor, buf_ctx->global_ctx);
}

अटल व्योम ggml_backend_webgpu_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                  स्थिर ggml_tensor *   tensor,
                                                  व्योम *                तथ्य,
                                                  माप_प्रकार                offset,
                                                  माप_प्रकार                size) {
    WEBGPU_CPU_PROFILE_योग_START(get_tensor);
    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << तथ्य
                                                              << ", " << offset << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    माप_प्रकार योग_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    माप_प्रकार local_offset = योग_offset % 4;
    अगर (local_offset != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        योग_offset -= local_offset;
    }

    माप_प्रकार final_आकार = size + local_offset;
    अगर (final_आकार % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_आकार += 4 - (final_आकार % 4);
    }

    std::lock_guard<std::recursive_mutex> lock(buf_ctx->global_ctx->mutex);

    अगर (buf_ctx->global_ctx->get_tensor_staging_buf == nullptr ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_आकार() < final_आकार) {
        // Create a new staging buffer if it doesn't exist or is too small
        अगर (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_बफर(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_आकार,
                                  wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_बफर_To_बफर(buf_ctx->बफर, योग_offset, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_आकार);
    wgpu::Command_बफर commands = encoder.Finish();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_बफर(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::MapMode::Read, 0, final_आकार);
    // Must specify size here since the staging buffer might be larger than the tensor size
    स्थिर व्योम * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_आकार);

    // Copy the data from the mapped range to the output buffer
    std::स_नकल(तथ्य, (स्थिर व्योम *) ((स्थिर अक्षर *) mapped_range + local_offset), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_योग_END(get_tensor, buf_ctx->global_ctx);
}

अटल व्योम ggml_backend_webgpu_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << बफर << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_योग_START(सफा);
    ggml_backend_webgpu_बफर_context * buf_ctx = (ggml_backend_webgpu_बफर_context *) बफर->context;
    ggml_backend_webgpu_बफर_memset(buf_ctx->global_ctx, buf_ctx->बफर, value, 0, बफर->size);
    WEBGPU_CPU_PROFILE_योग_END(सफा, buf_ctx->global_ctx);
}

अटल ggml_backend_बफर_i ggml_backend_webgpu_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_webgpu_बफर_get_base,
    /* .init_tensor     = */ शून्य,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ शून्य,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_बफर_clear,
    /* .reset           = */ शून्य,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

अटल स्थिर अक्षर * ggml_backend_webgpu_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    वापस ctx->device_name.c_str();
}

अटल ggml_backend_बफर_t ggml_backend_webgpu_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                          माप_प्रकार                     size) {
    अटल std::atomic<पूर्णांक> बफर_गिनती;
    पूर्णांक                     बफर_id = बफर_गिनती++;
    std::string             buf_name  = "tensor_buf" + std::to_string(बफर_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << बफर_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::बफर_2                         buf;
    ggml_webgpu_create_बफर(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::बफर_Usage::Storage | wgpu::बफर_Usage::CopySrc | wgpu::बफर_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_बफर_context * buf_ctx =
        नया ggml_backend_webgpu_बफर_context(buf, buf_name, ctx->webgpu_global_ctx);

    वापस ggml_backend_बफर_init(buft, ggml_backend_webgpu_बफर_interface, buf_ctx, size);
}

अटल माप_प्रकार ggml_backend_webgpu_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    वापस dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
अटल माप_प्रकार ggml_backend_webgpu_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    वापस dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार;
}

अटल माप_प्रकार ggml_backend_webgpu_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft,
                                                             स्थिर ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    माप_प्रकार                               res = ggml_nbytes(tensor);
    चयन (tensor->op) {
        हाल GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            अवरोध;
        हाल GGML_OP_शीर्ष_K:
            {
                स्थिर ggml_tensor * src0 = tensor->src[0];
                अगर (src0) {
                    स्थिर माप_प्रकार full = माप(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        full * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            {
                स्थिर ggml_tensor * Q            = tensor->src[0];
                स्थिर ggml_tensor * K            = tensor->src[1];
                स्थिर ggml_tensor * V            = tensor->src[2];
                स्थिर ggml_tensor * mask         = tensor->src[3];
                स्थिर स्वतः &        capabilities = ctx->webgpu_global_ctx->capabilities;
                अगर (ggml_webgpu_flash_attn_use_vec_path(ctx->webgpu_global_ctx, Q, K, V)) {
                    स्थिर bool kv_direct =
                        ggml_webgpu_flash_attn_k_direct(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH) ||
                        ggml_webgpu_flash_attn_v_direct(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH);
                    स्थिर uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_आकार, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        mask != nullptr, kv_direct);

                    स्थिर uint32_t vec_nwg_cap = capabilities.min_subgroup_आकार;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    स्थिर माप_प्रकार   align = capabilities.limits.min_Storage_बफर_Offset_Alignment;
                    स्थिर uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    अगर (nwg > 1u) {
                        स्थिर uint64_t tmp_तथ्य_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        स्थिर uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        स्थिर माप_प्रकार   tmp_आकार_bytes = ROUNDUP_POW2((tmp_तथ्य_elems + tmp_stats_elems) * माप(भग्न),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_आकार_bytes + align;
                    } अन्यथा {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    अगर (mask != nullptr) {
                        स्थिर uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        स्थिर uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        स्थिर uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_आकार(mask->type));
                        स्थिर uint32_t blk_batch_गिनती = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        स्थिर uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_गिनती;
                        स्थिर माप_प्रकार   blk_आकार_bytes =
                            ROUNDUP_POW2(blk_elems * माप(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_आकार_bytes + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            अवरोध;
        हाल GGML_OP_MUL_MAT:
            {
                स्थिर ggml_tensor * src0 = tensor->src[0];
                स्थिर ggml_tensor * src1 = tensor->src[1];
                bool                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_product,
                                             ctx->webgpu_global_ctx->vendor);
                अगर (use_mmvq) {
                    स्थिर माप_प्रकार q8_src1_आकार = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_आकार +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            अवरोध;
        हाल GGML_OP_MUL_MAT_ID:
            {
                स्थिर ggml_tensor * src0 = tensor->src[0];
                स्थिर ggml_tensor * src1 = tensor->src[1];
                अगर (src0 && src1) {
                    स्थिर माप_प्रकार ids_cache_आकार = माप(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    स्थिर माप_प्रकार ids_tiles_आकार = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_आकार + ids_tiles_आकार +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_बफर_Offset_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            अवरोध;
        शेष:
            अवरोध;
    }
    वापस res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

अटल स्थिर अक्षर * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    वापस ctx->device_name.c_str();
}

अटल स्थिर अक्षर * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    वापस ctx->device_desc.c_str();
}

अटल व्योम ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_बफर_आकार = ctx->webgpu_global_ctx->capabilities.limits.max_बफर_आकार_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#अगर UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_आकार = static_cast<uint64_t>(UINTPTR_MAX);
    अगर (max_बफर_आकार > max_ptr_आकार) {
        max_बफर_आकार = max_ptr_आकार;
    }
#पूर्ण_अगर
    *मुक्त  = static_cast<माप_प्रकार>(max_बफर_आकार);
    *योग = static_cast<माप_प्रकार>(max_बफर_आकार);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    वापस GGML_BACKEND_DEVICE_TYPE_GPU;
}

अटल व्योम ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

अटल ggml_guid_t ggml_backend_webgpu_guid(व्योम) {
    अटल ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    वापस &guid;
}

अटल व्योम ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    माप_प्रकार max_threads = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup *
                         ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_bytes_per_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_बफर_Binding_आकार, max_threads);
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].कुंजी     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    constants[1].कुंजी     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

अटल व्योम ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#अगर !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    स्थिर अक्षर * स्थिर          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_गिनती = 2;
    options.अग्र_In_Chain                   = &adapterTogglesDesc;
#पूर्ण_अगर

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter _adapter, wgpu::StringView message) {
                अगर (status != wgpu::RequestAdapterStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(message).c_str());
                    वापस;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

अटल व्योम create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != nullptr);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#अगर !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    अगर (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.अग्र_In_Chain = &subgroup_matrix_configs;
    }
#पूर्ण_अगर
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_आकार = ggml_backend_webgpu_get_command_submit_batch_आकार();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_product = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8IntegerDotProduct);

    bool valid_subgroup_matrix_config = false;
#अगर !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    अगर (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        क्रम (माप_प्रकार i = 0; i < subgroup_matrix_configs.config_गिनती; i++) {
            स्थिर wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            अगर (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.परिणाम_Component_Type == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = true;
                अवरोध;
            }
        }
    }
#पूर्ण_अगर
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_आकार = info.subgroup_Min_आकार;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_आकार = info.subgroup_Max_आकार;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#अगर !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    अगर (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#पूर्ण_अगर

    अगर (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#पूर्ण_अगर

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.तथ्य();
    dev_desc.required_Feature_गिनती = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](स्थिर wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            अगर (reason == wgpu::DeviceLostReason::Destroyed) {
                वापस;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<पूर्णांक>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](स्थिर wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<पूर्णांक>(reason),
                       std::string(message).c_str());
        });

#अगर !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    स्थिर अक्षर * स्थिर          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    स्थिर अक्षर * स्थिर          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_गिनती  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_गिनती = 1;

    dev_desc.अग्र_In_Chain = &deviceTogglesDesc;
#पूर्ण_अगर

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                अगर (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                    वापस;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_बफर(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_आकार_BYTES, wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.GetQueue();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

अटल webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_आकार_BYTES,
        webgpu_ctx->global_ctx->command_submit_batch_आकार + WEBGPU_अंक_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_Uniform_बफर_Offset_Alignment);
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_आकार_BYTES,
                              wgpu::बफर_Usage::Storage | wgpu::बफर_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_host_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_आकार_BYTES,
                              wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "set_rows_host_error_buf");

#अगर_घोषित GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = false;
    ggml_webgpu_create_बफर(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_आकार_BYTES,
        wgpu::बफर_Usage::QueryResolve | wgpu::बफर_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_आकार_BYTES,
                              wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::QuerySetDescriptor query_set_desc = {};
    query_set_desc.type                     = wgpu::QueryType::Timestamp;
    query_set_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_गिनती;
    webgpu_ctx->profile_timestamp_query_set = webgpu_ctx->global_ctx->device.CreateQuerySet(&query_set_desc);
#पूर्ण_अगर

#अगर_घोषित GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * माप(uint32_t),
                              wgpu::बफर_Usage::CopyDst | wgpu::बफर_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_बफर(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * माप(uint32_t),
                              wgpu::बफर_Usage::Storage | wgpu::बफर_Usage::CopySrc, "debug_dev_buf");
#पूर्ण_अगर
    वापस webgpu_ctx;
}

अटल ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    स्वतः * backend_ctx      = नया ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    स्वतः * backend = नया ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    वापस backend;
}

अटल ggml_backend_बफर_type_t ggml_backend_webgpu_device_get_बफर_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    अटल काष्ठा ggml_backend_बफर_type ggml_backend_webgpu_बफर_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_बफर_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_बफर_type_alloc_बफर,
                        /* .alloc_buffer_n      = */ शून्य,
                        /* .get_alignment       = */ ggml_backend_webgpu_बफर_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_बफर_type_get_max_आकार,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_बफर_type_get_alloc_आकार,
                        /* .get_alloc_size_n    = */ शून्य,
                        /* .is_host             = */ शून्य,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ शून्य
    };

    वापस &ggml_backend_webgpu_बफर_type;
}

अटल bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    GGML_UNUSED(dev);
    वापस buft->iface.get_name == ggml_backend_webgpu_बफर_type_get_name;
}

अटल bool ggml_webgpu_supported_qtype(ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
            वापस true;
        शेष:
            वापस false;
    }
}

अटल bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    अगर (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार)) {
        वापस false;
    }

    bool supports_op = false;
    चयन (op->op) {
        हाल GGML_OP_NONE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_RESHAPE:
            supports_op = true;
            अवरोध;
        हाल GGML_OP_ADD:
        हाल GGML_OP_SUB:
        हाल GGML_OP_MUL:
        हाल GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            अवरोध;
        हाल GGML_OP_ADD_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            अवरोध;
        हाल GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            अवरोध;
        हाल GGML_OP_DUP:
        हाल GGML_OP_CPY:
        हाल GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            अवरोध;
        हाल GGML_OP_SET:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            अवरोध;
        हाल GGML_OP_SET_ROWS:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            अवरोध;
        हाल GGML_OP_GET_ROWS:
            अगर (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } अन्यथा अगर (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            अवरोध;
        हाल GGML_OP_MUL_MAT:
            {
                चयन (src1->type) {
                    हाल GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        अवरोध;
                    हाल GGML_TYPE_F32:
                        चयन (src0->type) {
                            हाल GGML_TYPE_F32:
                            हाल GGML_TYPE_F16:
                            हाल GGML_TYPE_BF16:
                            हाल GGML_TYPE_Q1_0:
                            हाल GGML_TYPE_Q2_0:
                            हाल GGML_TYPE_Q4_0:
                            हाल GGML_TYPE_Q4_1:
                            हाल GGML_TYPE_Q5_0:
                            हाल GGML_TYPE_Q5_1:
                            हाल GGML_TYPE_Q8_0:
                            हाल GGML_TYPE_Q2_K:
                            हाल GGML_TYPE_Q3_K:
                            हाल GGML_TYPE_Q4_K:
                            हाल GGML_TYPE_Q5_K:
                            हाल GGML_TYPE_Q6_K:
                            हाल GGML_TYPE_IQ2_XXS:
                            हाल GGML_TYPE_IQ2_XS:
                            हाल GGML_TYPE_IQ2_S:
                            हाल GGML_TYPE_IQ3_XXS:
                            हाल GGML_TYPE_IQ3_S:
                            हाल GGML_TYPE_IQ1_S:
                            हाल GGML_TYPE_IQ1_M:
                            हाल GGML_TYPE_IQ4_NL:
                            हाल GGML_TYPE_IQ4_XS:
                            हाल GGML_TYPE_MXFP4:
                            हाल GGML_TYPE_NVFP4:
                                supports_op = true;
                                अवरोध;
                            शेष:
                                अवरोध;
                        }
                    शेष:
                        अवरोध;
                }
                अवरोध;
            }
        हाल GGML_OP_MUL_MAT_ID:
            चयन (src1->type) {
                हाल GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    अवरोध;
                हाल GGML_TYPE_F32:
                    चयन (src0->type) {
                        हाल GGML_TYPE_F32:
                        हाल GGML_TYPE_F16:
                        हाल GGML_TYPE_BF16:
                        हाल GGML_TYPE_Q1_0:
                        हाल GGML_TYPE_Q4_0:
                        हाल GGML_TYPE_Q4_1:
                        हाल GGML_TYPE_Q5_0:
                        हाल GGML_TYPE_Q5_1:
                        हाल GGML_TYPE_Q8_0:
                        हाल GGML_TYPE_Q2_K:
                        हाल GGML_TYPE_Q3_K:
                        हाल GGML_TYPE_Q4_K:
                        हाल GGML_TYPE_Q5_K:
                        हाल GGML_TYPE_Q6_K:
                        हाल GGML_TYPE_IQ1_S:
                        हाल GGML_TYPE_IQ1_M:
                        हाल GGML_TYPE_IQ2_XXS:
                        हाल GGML_TYPE_IQ2_XS:
                        हाल GGML_TYPE_IQ2_S:
                        हाल GGML_TYPE_IQ3_XXS:
                        हाल GGML_TYPE_IQ3_S:
                        हाल GGML_TYPE_IQ4_NL:
                        हाल GGML_TYPE_IQ4_XS:
                        हाल GGML_TYPE_MXFP4:
                        हाल GGML_TYPE_NVFP4:
                            supports_op = true;
                            अवरोध;
                        शेष:
                            अवरोध;
                    }
                    अवरोध;
                शेष:
                    अवरोध;
            }
            अगर (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = false;
            }
            अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                अगर (!supports_op) {
                    अवरोध;
                }
                अगर (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = false;
                    अवरोध;
                }
                स्थिर स्वतः & capabilities             = ctx->webgpu_global_ctx->capabilities;
                स्थिर माप_प्रकार storage_offset_alignment = capabilities.limits.min_Storage_बफर_Offset_Alignment;

                // subgroup matrix path requirements
                स्थिर bool use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
                    capabilities.supports_subgroup_matrix, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                स्थिर bool float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_offset_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_offset_alignment));
                स्थिर uint32_t k_tile_शीर्ष_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_आकार(src1->type);
                स्थिर uint32_t v_tile_शीर्ष_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_आकार(src2->type);
                स्थिर bool     tile_kv_शीर्ष_dims_aligned =
                    src0->ne[0] % k_tile_शीर्ष_align == 0 && src2->ne[0] % v_tile_शीर्ष_align == 0;
                स्थिर bool tile_can_dispatch_all_q_rows =
                    capabilities.limits.maxComputeInvocationsPerWorkgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_आकार;
                स्थिर bool use_tile = !use_subgroup_matrix && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_शीर्ष_dims_aligned && tile_can_dispatch_all_q_rows;

                अगर (!use_subgroup_matrix && !use_tile) {
                    supports_op = false;
                    अवरोध;
                }
                स्थिर uint32_t q_tile =
                    use_subgroup_matrix ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                स्थिर uint32_t kv_granularity = use_subgroup_matrix ? capabilities.sg_mat_n : 1u;
                स्थिर bool     kv_direct = use_subgroup_matrix ?
                                               ggml_webgpu_flash_attn_k_direct(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_direct(src0, src2, capabilities.sg_mat_k) :
                                               false;
                स्थिर uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_आकार, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != nullptr, kv_direct);
                supports_op = max_kv_tile > 0;
                अवरोध;
            }
        हाल GGML_OP_RMS_NORM:
        हाल GGML_OP_NORM:
        हाल GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_rows(src0);
            अवरोध;
        हाल GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            अवरोध;
        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(op)) {
                हाल GGML_GLU_OP_REGLU:
                हाल GGML_GLU_OP_GEGLU:
                हाल GGML_GLU_OP_SWIGLU:
                हाल GGML_GLU_OP_GEGLU_ERF:
                हाल GGML_GLU_OP_GEGLU_QUICK:
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    अवरोध;
                हाल GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    अवरोध;
                शेष:
                    अवरोध;
            }
            अवरोध;
        हाल GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_UNARY:
            {
                स्थिर ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                चयन (UNARY_OP) {
                    हाल GGML_UNARY_OP_ABS:
                    हाल GGML_UNARY_OP_SGN:
                    हाल GGML_UNARY_OP_NEG:
                    हाल GGML_UNARY_OP_STEP:
                    हाल GGML_UNARY_OP_TANH:
                    हाल GGML_UNARY_OP_ELU:
                    हाल GGML_UNARY_OP_RELU:
                    हाल GGML_UNARY_OP_SIGMOID:
                    हाल GGML_UNARY_OP_GELU:
                    हाल GGML_UNARY_OP_GELU_QUICK:
                    हाल GGML_UNARY_OP_SILU:
                    हाल GGML_UNARY_OP_HARDSWISH:
                    हाल GGML_UNARY_OP_HARDSIGMOID:
                    हाल GGML_UNARY_OP_EXP:
                    हाल GGML_UNARY_OP_GELU_ERF:
                    हाल GGML_UNARY_OP_SOFTPLUS:
                    हाल GGML_UNARY_OP_EXPM1:
                    हाल GGML_UNARY_OP_FLOOR:
                    हाल GGML_UNARY_OP_CEIL:
                    हाल GGML_UNARY_OP_ROUND:
                    हाल GGML_UNARY_OP_TRUNC:
                    हाल GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        अवरोध;
                    शेष:
                        अवरोध;
                }
            }
            अवरोध;
        हाल GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            अवरोध;
        हाल GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            अवरोध;
        हाल GGML_OP_IM2COL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            अवरोध;
        हाल GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_SSM_SCAN:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            अवरोध;
        हाल GGML_OP_GATED_DELTA_NET:
            {
                स्थिर uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            }
            अवरोध;
        हाल GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            अवरोध;
        हाल GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            अवरोध;
        हाल GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            अवरोध;
        हाल GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            अवरोध;
        हाल GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            अवरोध;
        हाल GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            अवरोध;
        हाल GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            अवरोध;
        हाल GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            अवरोध;
        हाल GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            अवरोध;
        हाल GGML_OP_शीर्ष_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            अवरोध;
        हाल GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            अवरोध;
        हाल GGML_OP_योग:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            अवरोध;
        हाल GGML_OP_योग_ROWS:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
            अवरोध;
        हाल GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            अवरोध;
        शेष:
            अवरोध;
    }
    अगर (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार) ||
        (src2 != nullptr &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_बफर_Binding_आकार)) {
        supports_op = false;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    अगर (!supports_op) {
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
    वापस supports_op;
}

अटल काष्ठा ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_बफर_type,
    /* .get_host_buffer_type = */ शून्य,
    /* .buffer_from_host_ptr = */ शून्य,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ शून्य,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

अटल स्थिर अक्षर * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    वापस ctx->name;
}

अटल माप_प्रकार ggml_backend_webgpu_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    वापस ctx->device_गिनती;
}

// Only one device is supported for now
अटल ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_योग_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    अटल ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    अटल ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_योग_END(reg_get_device, reg_ctx->webgpu_global_ctx);
    वापस &device;
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ शून्य,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    अटल ggml_backend_webgpu_reg_context * ctx = नया ggml_backend_webgpu_reg_context();

    अटल ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_गिनती = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    अगर (ctx->webgpu_global_ctx != nullptr && ctx->webgpu_global_ctx->instance != nullptr) {
        वापस &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.तथ्य();
    instance_descriptor.required_Feature_गिनती                 = instance_features.size();

#अगर !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    स्थिर अक्षर * स्थिर          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_गिनती = 1;
    instance_descriptor.अग्र_In_Chain        = &instanceTogglesDesc;
#पूर्ण_अगर

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(नया webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    अगर (ctx->webgpu_global_ctx->instance != nullptr) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    अगर (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#अगर !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#पूर्ण_अगर
    ) {
        ctx->device_गिनती = 1;
    }

    वापस &reg;
}

ggml_backend_t ggml_backend_webgpu_init(व्योम) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    अगर (ggml_backend_reg_dev_गिनती(reg) == 0) {
        वापस nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    वापस ggml_backend_webgpu_backend_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
