/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#अन्तर्भाव "ggml-webgpu.h"

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-webgpu-shader-lib.hpp"
#अन्तर्भाव "ggml.h"

#ifdef __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#endif

#अन्तर्भाव <webgpu/webgpu_cpp.h>

#अन्तर्भाव <atomic>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstring>
#ifdef GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#endif
#सचे defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#endif
#अन्तर्भाव <memory>
#अन्तर्भाव <mutex>
#अन्तर्भाव <optional>
#अन्तर्भाव <string>
#अन्तर्भाव <utility>
#अन्तर्भाव <vector>

#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
स्थिर अन्तःस्थ शून्य compute_2d_workgroups(uint32_t total_wg, uint32_t max_per_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(total_wg, max_per_dim));
    wg_x = CEIL_DIV(total_wg, wg_y);
}

स्थिर अन्तःस्थ uint32_t ggml_webgpu_u32_from_f32(भिन्न value) {
    uint32_t bits;
    memcpy(&bits, &value, परिमाण(bits));
    पटिनिवत्त bits;
}

#ifdef GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(सन्देशः)  std::cout << सन्देशः << std::endl
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#अञ्ञथा
#    define WEBGPU_LOG_DEBUG(सन्देशः) ((शून्य) 0)
#endif  // GGML_WEBGPU_DEBUG

#ifdef GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id) auto cpu_total_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)                                                         \
        auto   cpu_total_end_##id = std::chrono::high_resolution_clock::now();                            \
        द्विगुण cpu_total_time_##id =                                                                      \
            std::chrono::duration<द्विगुण, std::milli>(cpu_total_end_##id - cpu_total_start_##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_total_time_##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) auto cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)                                                          \
        auto   cpu_detail_end_##id = std::chrono::high_resolution_clock::now();                             \
        द्विगुण cpu_detail_time_##id =                                                                       \
            std::chrono::duration<द्विगुण, std::milli>(cpu_detail_end_##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#अञ्ञथा
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id)
#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)
#endif  // GGML_WEBGPU_CPU_PROFILE

#ifdef GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_गणनम्        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES (WEBGPU_MAX_PROFILE_QUERY_गणनम् * परिमाण(uint64_t))
#endif

/* Constants */

#define WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE 64u
#define WEBGPU_अङ्कः_PARAM_SLOT_SAFETY_MARGIN      10u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#define WEBGPU_PARAMS_BUF_SIZE_BYTES             128  // enough for 32 parameters
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES     4
#define WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
स्थिर शून्य * ध्रुव webgpu_ptr_base = (शून्य *) (uintptr_t) 0x1000;  // NOLINT

स्थिर size_t ggml_webgpu_tensor_offset(ध्रुव ggml_tensor * tensor) {
    ध्रुव ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    पटिनिवत्त (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
स्थिर शून्य ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      ध्रुव वर्ण *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
संरचना webgpu_param_arena {
    wgpu::Buffer buffer;
    size_t       slot_stride = 0;
    size_t       slot_size   = 0;
    uint32_t     slot_गणनम्  = 0;
    uint32_t     next_slot   = 0;

    शून्य init(wgpu::Device device, size_t slot_size, uint32_t slot_गणनम्, size_t alignment) {
        एतत्->slot_stride = ROUNDUP_POW2(slot_size, alignment);
        एतत्->slot_size   = slot_size;
        एतत्->slot_गणनम्  = slot_गणनम्;
        एतत्->next_slot   = 0;

        ggml_webgpu_create_buffer(device, buffer, एतत्->slot_stride * slot_गणनम्,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_slot(size_t size) {
        GGML_ASSERT(size <= slot_size);
        सचे (next_slot >= slot_गणनम्) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        पटिनिवत्त slot_stride * next_slot++;
    }

    शून्य reset() { next_slot = 0; }

    शून्य cleanup() {
        सचे (buffer) {
            buffer.Destroy();
            buffer = nullptr;
        }
    }

    ~webgpu_param_arena() { एतत्->cleanup(); }
};

संरचना webgpu_encoded_op {
    uint32_t अङ्कः_kernels = 0;
#ifdef GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_names;
#endif
};

संरचना webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

संरचना webgpu_capabilities {
    wgpu::Limits limits;
    तार्किक         supports_subgroups       = मिच्छा;
    तार्किक         supports_subgroup_matrix = मिच्छा;
    तार्किक         supports_dot_product     = मिच्छा;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_size     = 0;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
    size_t   memset_bytes_per_thread;
};

// Stores global webgpu members
संरचना webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    uint32_t       command_submit_batch_size = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::Buffer         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::Buffer    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#ifdef GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, द्विगुण> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, द्विगुण> cpu_detail_ms;
#endif

#ifdef GGML_WEBGPU_DEBUG
    wgpu::Buffer debug_host_buf;
    wgpu::Buffer debug_dev_buf;
#endif

    ~webgpu_global_context_struct() {
        सचे (एतत्->get_tensor_staging_buf) {
            एतत्->get_tensor_staging_buf.Destroy();
            एतत्->get_tensor_staging_buf = nullptr;
        }
        सचे (एतत्->memset_params_buf) {
            एतत्->memset_params_buf.Destroy();
            एतत्->memset_params_buf = nullptr;
        }
#ifdef GGML_WEBGPU_DEBUG
        सचे (एतत्->debug_host_buf) {
            एतत्->debug_host_buf.Destroy();
            एतत्->debug_host_buf = nullptr;
        }
        सचे (एतत्->debug_dev_buf) {
            एतत्->debug_dev_buf.Destroy();
            एतत्->debug_dev_buf = nullptr;
        }
#endif
    }
};

प्रकारनिर्देश std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
संरचना webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::Buffer             set_rows_dev_error_buf;
    wgpu::Buffer             set_rows_host_error_buf;
    wgpu::CommandEncoder     active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;
    तार्किक                     batch_compute_passes = सच्च;

    size_t memset_bytes_per_thread;

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, द्विगुण> shader_gpu_time_ms;
    wgpu::Buffer                            profile_timestamp_dev_buf;
    wgpu::Buffer                            profile_timestamp_host_buf;
    wgpu::QuerySet                          profile_timestamp_query_set;
    uint32_t                                profile_timestamp_query_गणनम् = 0;
#endif

    ~webgpu_context_struct() {
#ifdef GGML_WEBGPU_GPU_PROFILE
        सचे (एतत्->profile_timestamp_host_buf) {
            एतत्->profile_timestamp_host_buf.Destroy();
            एतत्->profile_timestamp_host_buf = nullptr;
        }
        सचे (एतत्->profile_timestamp_dev_buf) {
            एतत्->profile_timestamp_dev_buf.Destroy();
            एतत्->profile_timestamp_dev_buf = nullptr;
        }
        सचे (एतत्->profile_timestamp_query_set) {
            एतत्->profile_timestamp_query_set.Destroy();
            एतत्->profile_timestamp_query_set = nullptr;
        }
#endif
        सचे (एतत्->set_rows_host_error_buf) {
            एतत्->set_rows_host_error_buf.Destroy();
            एतत्->set_rows_host_error_buf = nullptr;
        }
        सचे (एतत्->set_rows_dev_error_buf) {
            एतत्->set_rows_dev_error_buf.Destroy();
            एतत्->set_rows_dev_error_buf = nullptr;
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
संरचना ggml_backend_webgpu_buffer_context {
    wgpu::Buffer          buffer;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_buffer_context(wgpu::Buffer buf, std::string lbl, webgpu_global_context global_ctx_) :
        buffer(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

स्थिर webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ध्रुव वर्ण *                             shader_code,
                                                   ध्रुव वर्ण *                             label,
                                                   ध्रुव std::vector<wgpu::ConstantEntry> & constants = {}) {
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.nextInChain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = nullptr;  // nullptr means auto layout
    सचे (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.data();
        pipeline_desc.compute.constant_गणनम् = constants.size();
    }
    पटिनिवत्त { device.CreateComputePipeline(&pipeline_desc), label };
}

स्थिर शून्य ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      ध्रुव वर्ण *      label) {
    wgpu::BufferDescriptor buffer_desc;
    buffer_desc.size             = size;
    buffer_desc.usage            = usage;
    buffer_desc.label            = label;
    buffer_desc.mappedAtCreation = मिच्छा;

    // TODO: error handling
    buffer = device.CreateBuffer(&buffer_desc);
}

स्थिर wgpu::Buffer ggml_webgpu_tensor_buf(ध्रुव ggml_tensor * tensor) {
    ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    पटिनिवत्त ctx->buffer;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
स्थिर size_t ggml_webgpu_tensor_align_offset(ध्रुव ggml_tensor * t, size_t alignment) {
    ध्रुव size_t offset    = ggml_webgpu_tensor_offset(t);
    ध्रुव size_t type_size = ggml_type_size(t->type);
    size_t       aligned   = offset & ~(alignment - 1);
    याव ((offset - aligned) % type_size != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    पटिनिवत्त aligned;
}

स्थिर size_t ggml_webgpu_tensor_misalignment(ध्रुव ggml_tensor * t, size_t alignment) {
    पटिनिवत्त ggml_webgpu_tensor_offset(t) - ggml_webgpu_tensor_align_offset(t, alignment);
}

स्थिर size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ध्रुव ggml_tensor * t) {
    पटिनिवत्त ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

स्थिर size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, ध्रुव ggml_tensor * t) {
    पटिनिवत्त ggml_webgpu_tensor_align_offset(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

स्थिर size_t ggml_webgpu_tensor_binding_size(ध्रुव ggml_tensor * t, size_t alignment) {
    पटिनिवत्त ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

स्थिर size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, ध्रुव ggml_tensor * t) {
    पटिनिवत्त ggml_webgpu_tensor_binding_size(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

स्थिर तार्किक ggml_webgpu_tensor_binding_overlap(ध्रुव webgpu_global_context & global_ctx,
                                               ध्रुव ggml_tensor *           a,
                                               ध्रुव ggml_tensor *           b) {
    सचे (a->buffer != b->buffer) {
        पटिनिवत्त मिच्छा;
    }

    ध्रुव size_t alignment = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    ध्रुव size_t a_offset  = ggml_webgpu_tensor_align_offset(a, alignment);
    ध्रुव size_t b_offset  = ggml_webgpu_tensor_align_offset(b, alignment);
    पटिनिवत्त a_offset < b_offset + ggml_webgpu_tensor_binding_size(b, alignment) &&
           b_offset < a_offset + ggml_webgpu_tensor_binding_size(a, alignment);
}

स्थिर तार्किक ggml_webgpu_tensor_binding_overlap_range(ध्रुव webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_buffer_t         buffer,
                                                     size_t                        offset,
                                                     size_t                        size) {
    सचे (tensor->buffer != buffer) {
        पटिनिवत्त मिच्छा;
    }

    ध्रुव size_t alignment     = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    ध्रुव size_t tensor_offset = ggml_webgpu_tensor_align_offset(tensor, alignment);
    पटिनिवत्त tensor_offset < offset + size && offset < tensor_offset + ggml_webgpu_tensor_binding_size(tensor, alignment);
}

संरचना ggml_webgpu_merged_binding_range {
    size_t offset;
    size_t size;
};

स्थिर ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_offset = SIZE_MAX;
    size_t merged_end    = 0;

    अत्थाय (ggml_tensor * tensor : tensors) {
        ध्रुव size_t bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
        ध्रुव size_t bind_end    = bind_offset + ggml_webgpu_tensor_binding_size(ctx, tensor);

        merged_offset = std::min(merged_offset, bind_offset);
        merged_end    = std::max(merged_end, bind_end);
    }

    पटिनिवत्त { merged_offset, merged_end - merged_offset };
}

स्थिर uint32_t ggml_webgpu_tensor_merged_element_offset(ध्रुव ggml_tensor *                      tensor,
                                                         ध्रुव ggml_webgpu_merged_binding_range & merged_range) {
    पटिनिवत्त (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_size(tensor->type));
}

स्थिर wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
                                                              wgpu::Buffer buffer,
                                                              uint64_t     offset,
                                                              uint64_t     size) {
    wgpu::BindGroupEntry entry = {};
    entry.binding              = binding;
    entry.buffer               = std::move(buffer);
    entry.offset               = offset;
    entry.size                 = size;
    पटिनिवत्त entry;
}

स्थिर wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    पटिनिवत्त ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_offset(ctx, tensor),
                                             ggml_webgpu_tensor_binding_size(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

template <typename T>
स्थिर शून्य ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T                callback_status,
                                                  T                success_status,
                                                  ध्रुव वर्ण *     wait_name,
                                                  ध्रुव वर्ण *     failure_name,
                                                  ध्रुव वर्ण *     callback_message) {
    सचे (wait_status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    सचे (wait_status == wgpu::WaitStatus::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    सचे (callback_status != success_status) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<पूर्णाङ्क>(callback_status),
                   callback_message);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
स्थिर uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    पटिनिवत्त UINT32_MAX;
}

स्थिर uint32_t ggml_backend_webgpu_get_command_submit_batch_size() {
    पटिनिवत्त WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
}

स्थिर शून्य ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::QueueWorkDoneStatus callback_status = wgpu::QueueWorkDoneStatus::Error;
    std::string               callback_message;

    ध्रुव wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
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

स्थिर शून्य ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
                                           wgpu::Buffer &          buffer,
                                           wgpu::MapMode           mode,
                                           size_t                  offset,
                                           size_t                  size) {
    wgpu::MapAsyncStatus callback_status = wgpu::MapAsyncStatus::Error;
    std::string          callback_message;

    ध्रुव wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
                        [&callback_status, &callback_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                            callback_status  = status;
                            callback_message = std::string(message);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::MapAsyncStatus::Success,
                                          "Buffer map wait", "Buffer map", callback_message.c_str());
}

स्थिर शून्य ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ध्रुव wgpu::CommandBuffer commands,
                                                uint32_t &                अङ्कः_inflight_batches) {
    सचे (अङ्कः_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        अङ्कः_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    अङ्कः_inflight_batches++;
}

#ifdef GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
स्थिर शून्य ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());
    ध्रुव भिन्न * debug_data = (ध्रुव भिन्न *) ctx->debug_host_buf.GetConstMappedRange();
    std::cout << "debug[0]: " << debug_data[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#endif

स्थिर webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ध्रुव std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            result = {};
    std::vector<wgpu::BindGroup> bind_groups;
    std::vector<size_t>          param_offsets;
    result.अङ्कः_kernels = dispatches.size();

    अत्थाय (size_t i = 0; i < dispatches.size(); i++) {
        ध्रुव webgpu_dispatch_desc & dispatch     = dispatches[i];
        ध्रुव size_t                 param_size   = dispatch.params.size() * परिमाण(uint32_t);
        ध्रुव size_t                 param_offset = ctx->param_arena.alloc_slot(param_size);

        std::vector<wgpu::BindGroupEntry> entries            = dispatch.bind_group_entries;
        uint32_t                          params_binding_अङ्कः = entries.size();
        entries.push_back(ggml_webgpu_make_bind_group_entry(params_binding_अङ्कः, ctx->param_arena.buffer, param_offset,
                                                            ctx->param_arena.slot_size));

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = dispatch.pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entry_गणनम् = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.CreateBindGroup(&bind_group_desc));
        param_offsets.push_back(param_offset);
    }

    अत्थाय (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.WriteBuffer(ctx->param_arena.buffer, param_offsets[i], dispatches[i].params.data(),
                                           dispatches[i].params.size() * परिमाण(uint32_t));
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    अत्थाय (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_गणनम् + 2 <= WEBGPU_MAX_PROFILE_QUERY_गणनम्);
        ध्रुव uint32_t query_begin = ctx->profile_timestamp_query_गणनम्++;
        ध्रुव uint32_t query_end   = ctx->profile_timestamp_query_गणनम्++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.querySet                    = ctx->profile_timestamp_query_set;
        ts_writes.beginningOfPassWriteIndex   = query_begin;
        ts_writes.endOfPassWriteIndex         = query_end;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass(&pass_desc);

        pass.SetPipeline(dispatches[i].pipeline.pipeline);
        pass.SetBindGroup(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
        pass.End();
        result.pipeline_names.push_back(dispatches[i].pipeline.name);
    }
#अञ्ञथा
    अत्थाय (size_t i = 0; i < dispatches.size(); i++) {
        सचे (ctx->batch_compute_passes) {
            ctx->active_compute_pass.SetPipeline(dispatches[i].pipeline.pipeline);
            ctx->active_compute_pass.SetBindGroup(0, bind_groups[i]);
            ctx->active_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second,
                                                        1);
        } अञ्ञथा {
            wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass();
            pass.SetPipeline(dispatches[i].pipeline.pipeline);
            pass.SetBindGroup(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
            pass.End();
        }
    }
#endif

    पटिनिवत्त result;
}

स्थिर webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    पटिनिवत्त ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_group_entries), { wg_x, wg_y } },
    });
}

स्थिर शून्य ggml_backend_webgpu_buffer_memset(webgpu_global_context & ctx,
                                              wgpu::Buffer &          buf,
                                              uint32_t                value,
                                              size_t                  offset,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) offset, (uint32_t) size, value };
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_bind_group_entry(0, buf, 0, buf.GetSize()) };
    size_t                            bytes_per_wg =
        ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup * ctx->capabilities.memset_bytes_per_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, bytes_per_wg);

    ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * परिमाण(uint32_t));

    wgpu::BindGroupEntry params_entry = {};
    params_entry.binding              = 1;
    params_entry.buffer               = ctx->memset_params_buf;
    params_entry.offset               = 0;
    params_entry.size                 = WEBGPU_PARAMS_BUF_SIZE_BYTES;
    entries.push_back(params_entry);

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
    bind_group_desc.entry_गणनम् = entries.size();
    bind_group_desc.entries    = entries.data();
    bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.SetPipeline(ctx->memset_pipeline.pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.End();

    wgpu::CommandBuffer              command  = encoder.Finish();
    std::vector<wgpu::CommandBuffer> commands = { command };
    ctx->queue.Submit(commands.size(), commands.data());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    पटिनिवत्त ctx->name.c_str();
}

स्थिर शून्य ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ifdef GGML_WEBGPU_CPU_PROFILE
    std::cout << "\n[ggml_webgpu cpu profiling summary]\n";
    द्विगुण total_cpu = 0.0;
    अत्थाय (ध्रुव auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        total_cpu += kv.second;
    }
    std::cout << "ggml_webgpu: total cpu time: " << total_cpu << " ms\n";
    std::cout << "ggml_webgpu: cpu breakdown:\n";
    अत्थाय (ध्रुव auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        द्विगुण pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    सचे (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::cout << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    अत्थाय (ध्रुव auto & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        द्विगुण pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    std::cout << "\n[ggml_webgpu gpu profiling summary]\n";
    द्विगुण total_gpu = 0.0;
    अत्थाय (ध्रुव auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        total_gpu += kv.second;
    }
    std::cout << "ggml_webgpu: total gpu time (all shaders): " << total_gpu << " ms\n";
    std::cout << "\nggml_webgpu: gpu breakdown:\n";
    अत्थाय (ध्रुव auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        द्विगुण pct = (total_gpu > 0.0) ? (kv.second / total_gpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#endif

#सचे defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::cout << "ggml_webgpu: gpu/cpu ratio: " << (total_cpu > 0.0 ? total_gpu / total_cpu : 0.0) << "\n";
#endif

    delete ctx;
    delete backend;
}

स्थिर webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)), (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)), (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
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
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ध्रुव तार्किक inplace   = decisions->inplace;

    ध्रुव uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    ध्रुव uint32_t dst_type_size = (uint32_t) ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[3] / dst_type_size),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        1u,
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[0] / dst_type_size),
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[1] / dst_type_size),
        (uint32_t) (((ध्रुव int32_t *) dst->op_params)[2] / dst_type_size),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          binding_index = 0;
    सचे (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        binding_index++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ध्रुव uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
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
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ध्रुव uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_size);
    ध्रुव uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
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
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
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
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ध्रुव तार्किक      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
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

    ध्रुव uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_size(src1->type));
    ध्रुव uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0;
    ध्रुव uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_size(src1->type));
    ध्रुव uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_size(src1->type));

    ध्रुव uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_size(dst->type));
    ध्रुव uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_size(dst->type));
    ध्रुव uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)) : 0;
    ध्रुव uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_size(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ध्रुव uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_per_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),

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

    ध्रुव uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_size);
    ध्रुव uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    सचे (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    सचे (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    सचे (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_scan_pipeline(shader_lib_ctx, xbc_overlap);
    auto *          decisions = static_cast<ggml_webgpu_ssm_scan_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t offset_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    uint32_t offset_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type));
    uint32_t offset_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_size(src3->type));
    uint32_t offset_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_size(src4->type));
    uint32_t offset_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_size(src5->type));
    uint32_t offset_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_size(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    सचे (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
    } अञ्ञथा सचे (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    } अञ्ञथा सचे (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_offset = 0;
    size_t xbc_bind_size   = 0;
    सचे (xbc_overlap > 0) {
        xbc_bind_offset = xbc_merged_range.offset;
        xbc_bind_size   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        offset_x,
        offset_dt,
        offset_A,
        offset_B,
        offset_C,
        offset_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_size(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_size(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_size(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_अङ्कः = 0;

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, src0),
    };
    // xbc_merged binding
    सचे (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_अङ्कः++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_offset, xbc_bind_size));
    }
    // x
    सचे (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, src1));
    }
    // B
    सचे (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, src4));
    }
    // C
    सचे (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः++, dst));

    ध्रुव uint32_t total_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ध्रुव uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ध्रुव uint32_t s_v      = (uint32_t) src2->ne[0];
    ध्रुव uint32_t h        = (uint32_t) src2->ne[1];
    ध्रुव uint32_t n_tokens = (uint32_t) src2->ne[2];
    ध्रुव uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ध्रुव uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ध्रुव भिन्न    scale    = 1.0f / sqrtf((भिन्न) s_v);
    uint32_t       scale_u32;
    memcpy(&scale_u32, &scale, परिमाण(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_tokens,
        n_seqs,
        s_v * h * n_tokens * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_size(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_size(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_size(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0), ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2), ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4), ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, dst),
    };

    सचे (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst_fuse));
    }

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

स्थिर std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    सचे (ggml_is_empty(src) || ggml_is_empty(idx)) {
        पटिनिवत्त std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_rows_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_set_rows_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)), (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)), (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::BindGroupEntry> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));

    सचे (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->set_rows_dev_error_buf, 0,
                                                            ctx->set_rows_dev_error_buf.GetSize()));
    }

    uint32_t threads;
    सचे (ggml_is_quantized(dst->type)) {
        ध्रुव uint32_t blocks_per_row = src->ne[0] / ggml_blck_size(dst->type);
        threads =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_blocks ? (blocks_per_row / 2) : blocks_per_row);
    } अञ्ञथा सचे (decisions->vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } अञ्ञथा {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(threads, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
स्थिर std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_size_entry(uint32_t wg_size) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].कुञ्चिका   = "wg_size";
    constants[0].value = wg_size;
    पटिनिवत्त constants;
}

स्थिर webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ध्रुव तार्किक float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ध्रुव uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    ध्रुव uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    ध्रुव uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    ध्रुव uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    ध्रुव uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));

    ध्रुव तार्किक vec4_aligned = offset_src % 4 == 0 && offset_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_rows_pipeline(shader_lib_ctx, vec4_aligned);
    auto *          decisions = static_cast<ggml_webgpu_get_rows_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { offset_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_size(idx->type)),
                                     offset_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_size(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_size(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
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
    uint32_t total_rows     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t total_threads  = float_parallel ? blocks_per_row * total_rows : total_rows;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(total_threads, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ध्रुव size_t dst_offset           = ggml_webgpu_tensor_offset(dst);
    ध्रुव size_t q8_src1_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    ध्रुव size_t q8_src1_binding_size = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> q8_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_offset, q8_src1_binding_size)
    };

    auto q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_size     = q8_decisions->wg_size;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ध्रुव uint32_t wg_per_vec     = (src0->ne[0] / 4 + (q8_wg_size - 1)) / q8_wg_size;
    ध्रुव uint32_t q8_total_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_per_vec;
    ध्रुव uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(q8_total_wg, max_wg_per_dim, q8_wg_x, q8_wg_y);

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
    तार्किक use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_product,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_size              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_size        = ctx->global_ctx->capabilities.min_subgroup_size;
    shader_lib_ctx.max_subgroup_size        = ctx->global_ctx->capabilities.max_subgroup_size;
    shader_lib_ctx.supports_dot_product     = ctx->global_ctx->capabilities.supports_dot_product;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    ध्रुव तार्किक src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    सचे (use_mat_vec) {
        सचे (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } अञ्ञथा {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    सचे (src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { offset_src0,
                                     offset_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::BindGroupEntry> entries = {};
    सचे (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        auto & mmvq_qq8_entry = dispatches[0].bind_group_entries[1];
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_entry.offset,
                                                            mmvq_qq8_entry.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    } अञ्ञथा सचे (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अञ्ञथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ध्रुव uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    सचे (use_mat_vec) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
        uint32_t total_wg      = output_groups * batches;
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    } अञ्ञथा {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        सचे (decisions->use_subgroup_matrix) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_matrix_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_matrix_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } अञ्ञथा {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_size_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_size_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t total_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    पटिनिवत्त ggml_backend_webgpu_build_multi(ctx, dispatches);
}

स्थिर webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    ध्रुव uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ध्रुव uint32_t param_n_expert_used = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_size(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_size(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_size(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ध्रुव uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
    uint32_t       total_wg       = output_groups * param_n_expert_used;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
स्थिर uint32_t ggml_webgpu_mul_mat_id_max_tiles(ध्रुव ggml_tensor * dst) {
    ध्रुव uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    ध्रुव uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    पटिनिवत्त CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
स्थिर size_t ggml_webgpu_mul_mat_id_tiles_nbytes(ध्रुव ggml_tensor * dst) {
    पटिनिवत्त (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * परिमाण(uint32_t);
}

स्थिर webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    सचे (dst->ne[2] == 1) {
        पटिनिवत्त ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ध्रुव uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ध्रुव uint32_t param_n_expert_used = (uint32_t) dst->ne[1];
    ध्रुव uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
    };

    ध्रुव size_t dst_offset       = ggml_webgpu_tensor_offset(dst);
    ध्रुव size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * परिमाण(uint32_t);
    ध्रुव size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ध्रुव size_t ids_cache_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    ध्रुव size_t ids_tiles_align_offset =
        ROUNDUP_POW2(ids_cache_align_offset + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);

    ध्रुव size_t ids_cache_bind_size = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ध्रुव size_t ids_tiles_bind_size = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::BindGroupEntry> gather_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_size(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_size),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_size),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::BindGroupEntry> main_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_size(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_size(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_size(ctx, dst)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_size),
        ggml_webgpu_make_bind_group_entry(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_size),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_size_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N);

    ध्रुव uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_size_m;
    ध्रुव uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ध्रुव uint32_t total_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    पटिनिवत्त ggml_backend_webgpu_build_multi(ctx, dispatches);
}

संरचना ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> entries;
    size_t                            kv_bind_offset = 0;
    size_t                            kv_bind_size   = 0;
    तार्किक                              has_mask       = मिच्छा;
    तार्किक                              has_sinks      = मिच्छा;
    तार्किक                              kv_overlap     = मिच्छा;
};

स्थिर तार्किक ggml_webgpu_flash_attn_use_vec_path(ध्रुव webgpu_global_context & global_ctx,
                                                ध्रुव ggml_tensor *           Q,
                                                ध्रुव ggml_tensor *           K,
                                                ध्रुव ggml_tensor *           V) {
    ध्रुव size_t storage_offset_alignment = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;

    ध्रुव तार्किक k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment);
    ध्रुव तार्किक v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);

    ध्रुव uint32_t k_vec_head_align =
        ggml_is_quantized(K->type) ? ggml_blck_size(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    ध्रुव uint32_t v_vec_head_align =
        ggml_is_quantized(V->type) ? ggml_blck_size(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    ध्रुव तार्किक kv_vec_head_dims_aligned = Q->ne[0] % k_vec_head_align == 0 && V->ne[0] % v_vec_head_align == 0;

    पटिनिवत्त global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_head_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

स्थिर ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    mask,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    भिन्न scale         = ggml_get_op_params_f32(dst, 0);
    भिन्न max_bias      = ggml_get_op_params_f32(dst, 1);
    भिन्न logit_softcap = ggml_get_op_params_f32(dst, 2);
    सचे (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    भिन्न n_head_log2 = भिन्न(1u << (uint32_t) floor(log2(Q->ne[2])));
    भिन्न m0          = powf(2.0f, -(max_bias) / n_head_log2);
    भिन्न m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = mask;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    op.shader_lib_ctx.max_wg_size              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    op.shader_lib_ctx.wg_mem_limit_bytes       = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_size        = ctx->global_ctx->capabilities.min_subgroup_size;
    op.shader_lib_ctx.max_subgroup_size        = ctx->global_ctx->capabilities.max_subgroup_size;

    op.has_mask   = mask != nullptr;
    op.has_sinks  = sinks != nullptr;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t offset_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type));
    uint32_t offset_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type));
    सचे (op.kv_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_offset                                   = merged_range.offset;
        op.kv_bind_size                                     = merged_range.size;
        offset_k                                            = ggml_webgpu_tensor_merged_element_offset(K, merged_range);
        offset_v                                            = ggml_webgpu_tensor_merged_element_offset(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
        offset_k,
        offset_v,
        op.has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_size(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_size(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_size(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_size(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_mask ? (uint32_t) (mask->nb[3] / ggml_type_size(mask->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
    };
    सचे (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_size));
    } अञ्ञथा {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
    }
    uint32_t binding_index = op.kv_overlap ? 2u : 3u;
    सचे (op.has_mask) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, mask));
    }
    सचे (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, dst));

    पटिनिवत्त op;
}

स्थिर uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ध्रुव uint64_t kv_span = (uint64_t) kv_tile;
    याव ((2u * nwg * kv_span) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    पटिनिवत्त std::min(nwg, vec_nwg_cap);
}

स्थिर webgpu_encoded_op ggml_webgpu_flash_attn_direct(webgpu_context & ctx, ध्रुव ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_per_head = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_per_head * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

स्थिर webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             mask,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    auto *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::Buffer blk_buf         = {};
    uint64_t     blk_size_bytes  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_गणनम् = 0;

    ध्रुव uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_size;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ध्रुव uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ध्रुव तार्किक     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_base  = 0;
    uint64_t     tmp_size_bytes  = 0;
    wgpu::Buffer tmp_buf         = {};
    uint64_t     tmp_bind_offset = 0;
    uint64_t     tmp_bind_size   = 0;
    ध्रुव size_t align_bytes     = ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    ध्रुव size_t dst_offset      = ggml_webgpu_tensor_offset(dst);
    size_t       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);

    सचे (use_vec_reduce) {
        ध्रुव uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ध्रुव uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_base                 = tmp_data_elems;
        tmp_size_bytes =
            ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * परिमाण(भिन्न), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    } अञ्ञथा {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_size_bytes  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::BindGroupEntry> blk_entries;
    सचे (op.has_mask) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ध्रुव uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
        blk_batch_गणनम्             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ध्रुव uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_गणनम्;
        blk_size_bytes              = ROUNDUP_POW2(blk_elems * परिमाण(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        ध्रुव ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(mask),
                                              ggml_webgpu_tensor_align_offset(ctx, mask),
                                              ggml_webgpu_tensor_binding_size(ctx, mask)),
            ggml_webgpu_make_bind_group_entry(1, blk_buf, scratch_offset, blk_size_bytes),
        };
        scratch_offset = ROUNDUP_POW2(scratch_offset + blk_size_bytes, align_bytes);
    }

    std::vector<uint32_t> split_params = op.params;
    सचे (op.has_mask) {
        split_params.push_back(0u);                     // blk_base
        split_params.push_back(blk_nblk0);              // blk_nblk0
        split_params.push_back(blk_nblk1);              // blk_nblk1
    }
    split_params.push_back(0u);                         // tmp_data_base
    split_params.push_back((uint32_t) tmp_stats_base);  // tmp_stats_base
    split_params.push_back(nwg);                        // nwg

    std::vector<wgpu::BindGroupEntry> split_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_offset(ctx, Q),
                                          ggml_webgpu_tensor_binding_size(ctx, Q)),
    };
    सचे (op.kv_overlap) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_size));
    } अञ्ञथा {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
                                                                  ggml_webgpu_tensor_binding_size(ctx, K)));
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
                                                                  ggml_webgpu_tensor_binding_size(ctx, V)));
    }
    uint32_t split_binding_index = op.kv_overlap ? 2u : 3u;
    सचे (op.has_mask) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(mask),
                                                                  ggml_webgpu_tensor_align_offset(ctx, mask),
                                                                  ggml_webgpu_tensor_binding_size(ctx, mask)));
    }
    सचे (op.has_sinks) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_offset(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_size(ctx, sinks)));
    }
    सचे (op.has_mask) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(split_binding_index++, blk_buf, blk_entries[1].offset, blk_size_bytes));
    }
    split_entries.push_back(
        ggml_webgpu_make_bind_group_entry(split_binding_index++, tmp_buf, tmp_bind_offset, tmp_bind_size));
    split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_offset(ctx, dst),
                                                              ggml_webgpu_tensor_binding_size(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::BindGroupEntry> reduce_entries;
    सचे (use_vec_reduce) {
        ध्रुव uint32_t reduce_sg_size = ctx->global_ctx->capabilities.max_subgroup_size;
        ध्रुव uint32_t reduce_wg_size = std::max(
            reduce_sg_size,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_size,
                                          ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_size                    = reduce_wg_size;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_base,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_group_entry(0, tmp_buf, tmp_bind_offset, tmp_size_bytes),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                              ggml_webgpu_tensor_binding_size(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ध्रुव uint64_t split_wg_total = (uint64_t) wg_x * nwg;
    GGML_ASSERT(split_wg_total <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    सचे (op.has_mask) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_गणनम् }
        });
    }
    dispatches.push_back({
        pipeline, std::move(split_params), std::move(split_entries), { (uint32_t) split_wg_total, 1u }
    });
    सचे (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    पटिनिवत्त ggml_backend_webgpu_build_multi(ctx, dispatches);
}

स्थिर webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    mask,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, mask, sinks, dst);
    सचे (ggml_webgpu_flash_attn_use_vec_path(ctx->global_ctx, Q, K, V)) {
        पटिनिवत्त ggml_webgpu_flash_attn_vec(ctx, Q, K, V, mask, sinks, dst, std::move(op));
    }
    पटिनिवत्त ggml_webgpu_flash_attn_direct(ctx, op);
}

स्थिर webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    तार्किक is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ध्रुव तार्किक inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    सचे (is_unary) {
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
    } अञ्ञथा सचे (dst->op == GGML_OP_CLAMP) {
        भिन्न clamp_min = ggml_get_op_params_f32(dst, 0);
        भिन्न clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } अञ्ञथा सचे (dst->op == GGML_OP_FILL) {
        भिन्न fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, effective_src),
    };
    सचे (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_द्विमान_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ध्रुव तार्किक      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_द्विमान_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_द्विमान_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
    size_t src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    सचे (decisions->src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;

    सचे (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अञ्ञथा {
        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        सचे (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_add_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_add_id_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_size(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::BindGroupEntry> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));

    सचे (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       total_wg       = ggml_nrows(dst);
    ध्रुव uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    सचे (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        पटिनिवत्त {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ध्रुव तार्किक      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_द्विमान_shader_decisions *>(pipeline.context.get());

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    सचे (decisions->src_overlap) {
        सचे (ggml_nbytes(src0) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src1);
            merged_size   = ggml_webgpu_tensor_binding_size(ctx, src1);
            offset_src0   = 0;
            offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
        } अञ्ञथा सचे (ggml_nbytes(src1) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
            merged_size   = ggml_webgpu_tensor_binding_size(ctx, src0);
            offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
            offset_src1   = 0;
        } अञ्ञथा {
            ध्रुव ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_offset = merged_range.offset;
            merged_size   = merged_range.size;
            offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
            offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     offset_src0,
                                     offset_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::BindGroupEntry> entries = {};
    सचे (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अञ्ञथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_size(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    सचे (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } अञ्ञथा सचे (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } अञ्ञथा {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t offset_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_size(rn_src->type));
    uint32_t offset_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_size(mul_src->type));
    size_t merged_offset = 0;
    size_t merged_size   = 0;

    std::vector<uint32_t> params = {
        offset_rn_src,
        offset_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_size(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_size(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_size(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_size(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ध्रुव तार्किक      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    सचे (decisions->src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_offset  = merged_range.offset;
        merged_size    = merged_range.size;
        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
        params[0]      = offset_rn_src;
        params[1]      = offset_mul_src;
    }

    सचे (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
    } अञ्ञथा सचे (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } अञ्ञथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
    सचे (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ध्रुव तार्किक inplace         = decisions->inplace;
    ध्रुव पूर्णाङ्क  has_freq_factor = (src2 != nullptr);

    ध्रुव पूर्णाङ्क n_dims     = ((int32_t *) dst->op_params)[1];
    ध्रुव पूर्णाङ्क mode       = ((int32_t *) dst->op_params)[2];
    ध्रुव पूर्णाङ्क n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ध्रुव पूर्णाङ्क n_offs     = ((int32_t *) dst->op_params)[15];

    भिन्न freq_base;
    भिन्न freq_scale;
    भिन्न ext_factor;
    भिन्न attn_factor;
    भिन्न beta_fast;
    भिन्न beta_slow;
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, परिमाण(भिन्न));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, परिमाण(भिन्न));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, परिमाण(भिन्न));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, परिमाण(भिन्न));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, परिमाण(भिन्न));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, परिमाण(भिन्न));

    पूर्णाङ्क sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * परिमाण(पूर्णाङ्क));

    भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        src2 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
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
    सचे (has_freq_factor) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
    }
    सचे (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ध्रुव तार्किक      src_overlap = src1 != nullptr && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    auto * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ध्रुव पूर्णाङ्क split = (src1 != nullptr);

    uint32_t offset_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1 =
        src1 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0;
    size_t merged_offset = 0;
    size_t merged_size   = 0;
    सचे (decisions->src_overlap) {
        ध्रुव ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_size   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
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
    सचे (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
    } अञ्ञथा {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    }
    सचे (split && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };

    सचे (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ध्रुव तार्किक inplace     = decisions->inplace;
    ध्रुव पूर्णाङ्क  has_mask    = (src1 != nullptr);
    ध्रुव पूर्णाङ्क  has_sink    = (src2 != nullptr);
    भिन्न      max_bias    = ggml_get_op_params_f32(dst, 1);
    भिन्न      n_head_log2 = भिन्न(1u << (uint32_t) floor(log2(src0->ne[2])));
    भिन्न      m0          = powf(2.0f, -(max_bias) / n_head_log2);
    भिन्न      m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
        has_mask ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_mask ? (uint32_t) src1->ne[2] : 0,
        has_mask ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_head_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_bind_group_entry(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
        ggml_webgpu_tensor_binding_size(ctx, src0)) };
    uint32_t                          binding_अङ्कः = 1;
    सचे (has_mask) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_अङ्कः, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_offset(ctx, src1),
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        binding_अङ्कः++;
    }
    सचे (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः, src2));
        binding_अङ्कः++;
    }
    सचे (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_अङ्कः, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = ggml_nrows(dst);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

स्थिर webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    तार्किक is_शिखरम्_k = dst->op == GGML_OP_शिखरम्_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    auto * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    ध्रुव uint32_t src_ne0 = (uint32_t) src->ne[0];
    ध्रुव uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ध्रुव uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_size);
    ध्रुव uint32_t block_size =
        is_शिखरम्_k ? std::min(argsort_decisions->wg_size, (uint32_t) dst->ne[0]) : argsort_decisions->wg_size;
    uint32_t out_ne0 = src_ne0;
    सचे (is_शिखरम्_k) {
        सचे (npr > 1) {
            ध्रुव uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_size;
            out_ne0                  = (npr - 1) * block_size + std::min(last_tile, block_size);
        } अञ्ञथा {
            out_ne0 = block_size;
        }
    }

    uint32_t merge_len    = block_size;
    uint32_t merge_passes = 0;
    याव (merge_len < out_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    ध्रुव तार्किक start_in_tmp = (merge_passes % 2) == 1;

    ध्रुव size_t dst_offset = ggml_webgpu_tensor_offset(dst);
    ध्रुव size_t idx_nbytes = out_ne0 * ggml_nrows(dst) * परिमाण(int32_t);
    ध्रुव size_t tmp_offset =
        ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    ध्रुव size_t tmp_binding_size = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ध्रुव size_t dst_binding_size =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ध्रुव uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    ध्रुव uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    ध्रुव uint32_t offset_tmp  = 0;
    ध्रुव uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    ध्रुव uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    ध्रुव uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));
    ध्रुव uint32_t stride_idx1 = out_ne0;
    ध्रुव uint32_t stride_idx2 = out_ne0 * (uint32_t) dst->ne[1];
    ध्रुव uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ध्रुव uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    ध्रुव size_t   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    ध्रुव size_t   init_binding_size = start_in_tmp ? tmp_binding_size : dst_binding_size;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_size,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ध्रुव uint32_t total_wg_init  = npr * nrows;
    ध्रुव uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(total_wg_init, max_wg_per_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::BindGroupEntry> init_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), init_align_offset, init_binding_size)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    सचे (merge_passes == 0) {
        पटिनिवत्त ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    तार्किक     in_is_tmp = start_in_tmp;
    uint32_t len       = block_size;
    याव (len < out_ne0) {
        ध्रुव uint32_t nm = CEIL_DIV(out_ne0, 2 * len);

        ध्रुव तार्किक     out_is_tmp  = !in_is_tmp;
        ध्रुव uint32_t offset_in   = in_is_tmp ? offset_tmp : offset_dst;
        ध्रुव uint32_t offset_out  = out_is_tmp ? offset_tmp : offset_dst;
        ध्रुव size_t   align_in    = in_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        ध्रुव size_t   align_out   = out_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        ध्रुव size_t   size_in     = in_is_tmp ? tmp_binding_size : dst_binding_size;
        ध्रुव size_t   size_out    = out_is_tmp ? tmp_binding_size : dst_binding_size;
        ध्रुव uint32_t शिखरम्_k_out   = (is_शिखरम्_k && nm == 1) ? (uint32_t) dst->ne[0] : out_ne0;
        ध्रुव uint32_t stride_out1 = शिखरम्_k_out;
        ध्रुव uint32_t stride_out2 = शिखरम्_k_out * (uint32_t) dst->ne[1];
        ध्रुव uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

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
                                               शिखरम्_k_out,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::BindGroupEntry> merge_entries = {
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), align_in, size_in),
            ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), align_out, size_out)
        };

        uint32_t       wg_x_merge;
        uint32_t       wg_y_merge;
        ध्रुव uint32_t total_wg_merge = nm * nrows;
        compute_2d_workgroups(total_wg_merge, max_wg_per_dim, wg_x_merge, wg_y_merge);

        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    पटिनिवत्त ggml_backend_webgpu_build_multi(ctx, dispatches);
}

स्थिर webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        total_wg = ggml_nrows(dst);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    तार्किक                  total_sum = dst->op == GGML_OP_SUM;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     total_sum ? 0 : (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                     total_sum ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     total_sum ? 1 : (uint32_t) src->ne[1],
                                     total_sum ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_sum_rows_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = total_sum ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

स्थिर तार्किक ggml_webgpu_can_fuse_rms_norm_mul(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    सचे (!ggml_can_fuse(cgraph, ग्रन्थिः_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        पटिनिवत्त मिच्छा;
    }

    // additional constraints specific to this fusion
    ध्रुव ggml_tensor * rms_norm = cgraph->nodes[ग्रन्थिः_idx];
    ध्रुव ggml_tensor * mul      = cgraph->nodes[ग्रन्थिः_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    सचे (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        पटिनिवत्त मिच्छा;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    सचे (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        पटिनिवत्त मिच्छा;
    }
    // rms_norm shader assumes contiguous rows
    सचे (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        पटिनिवत्त मिच्छा;
    }

    पटिनिवत्त सच्च;
}

स्थिर तार्किक ggml_webgpu_can_fuse_gdn_cache(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, पूर्णाङ्क & अङ्कः_encoded_ops) {
    ध्रुव ggml_tensor * gdn = cgraph->nodes[ग्रन्थिः_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    सचे (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        पटिनिवत्त मिच्छा;
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
    ध्रुव size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ध्रुव ggml_tensor * cpy     = nullptr;
    पूर्णाङ्क                 cpy_idx = 0;
    अत्थाय (पूर्णाङ्क j = ग्रन्थिः_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        ध्रुव ggml_tensor * n = cgraph->nodes[j];
        सचे (ggml_op_is_empty(n->op) || ggml_is_empty(n)) {
            अनुवर्त;
        }
        सचे (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            पटिनिवत्त मिच्छा;
        }
        cpy     = n;
        cpy_idx = j;
    }
    सचे (cpy == nullptr) {
        पटिनिवत्त मिच्छा;
    }

    ध्रुव ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ध्रुव ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    सचे (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != tail_off ||
        !ggml_is_contiguous(cpy_src)) {
        पटिनिवत्त मिच्छा;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ध्रुव std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    सचे (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        पटिनिवत्त मिच्छा;
    }

    अङ्कः_encoded_ops = cpy_idx - ग्रन्थिः_idx + 1;

    पटिनिवत्त सच्च;
}

स्थिर webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ध्रुव uint32_t        mode_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_size(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_size(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),

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
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    पटिनिवत्त ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
स्थिर std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           पूर्णाङ्क            ग्रन्थिः_idx,
                                                           पूर्णाङ्क &          अङ्कः_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  ग्रन्थिः  = nodes[ग्रन्थिः_idx];

    सचे (ggml_is_empty(ग्रन्थिः)) {
        पटिनिवत्त std::nullopt;
    }
    सचे ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        पटिनिवत्त std::nullopt;
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
            पटिनिवत्त std::nullopt;
        पक्ष GGML_OP_DUP:
        पक्ष GGML_OP_CPY:
        पक्ष GGML_OP_CONT:
            पटिनिवत्त ggml_webgpu_cpy(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_SET:
            पटिनिवत्त ggml_webgpu_set(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_SET_ROWS:
            पटिनिवत्त ggml_webgpu_set_rows(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_GET_ROWS:
            पटिनिवत्त ggml_webgpu_get_rows(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_MUL_MAT:
            पटिनिवत्त ggml_webgpu_mul_mat(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_MUL_MAT_ID:
            पटिनिवत्त ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, ग्रन्थिः);
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            पटिनिवत्त ggml_webgpu_flash_attn(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः);
        पक्ष GGML_OP_ADD:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
            पटिनिवत्त ggml_webgpu_द्विमान_op(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_ADD_ID:
            पटिनिवत्त ggml_webgpu_add_id(ctx, src0, src1, src2, ग्रन्थिः);
        पक्ष GGML_OP_CONCAT:
            पटिनिवत्त ggml_webgpu_concat(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_REPEAT:
            पटिनिवत्त ggml_webgpu_repeat(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_RMS_NORM:
            सचे (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, ग्रन्थिः_idx)) {
                अङ्कः_encoded_ops        = 2;
                ggml_tensor * mul_ग्रन्थिः = nodes[ग्रन्थिः_idx + 1];
                पटिनिवत्त ggml_webgpu_rms_norm_mul(ctx, src0, ग्रन्थिः, mul_ग्रन्थिः->src[0], mul_ग्रन्थिः->src[1], mul_ग्रन्थिः);
            } अञ्ञथा {
                पटिनिवत्त ggml_webgpu_row_norm(ctx, src0, ग्रन्थिः);
            }
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_L2_NORM:
            पटिनिवत्त ggml_webgpu_row_norm(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_ROPE:
            पटिनिवत्त ggml_webgpu_rope(ctx, src0, src1, src2, ग्रन्थिः);
        पक्ष GGML_OP_GLU:
            पटिनिवत्त ggml_webgpu_glu(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_SCALE:
            पटिनिवत्त ggml_webgpu_scale(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_SOFT_MAX:
            पटिनिवत्त ggml_webgpu_soft_max(ctx, src0, src1, src2, ग्रन्थिः);
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
            पटिनिवत्त ggml_webgpu_unary_op(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_SOLVE_TRI:
            पटिनिवत्त ggml_webgpu_solve_tri(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_SSM_CONV:
            पटिनिवत्त ggml_webgpu_ssm_conv(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_SSM_SCAN:
            पटिनिवत्त ggml_webgpu_ssm_scan(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः->src[5], ग्रन्थिः->src[6],
                                        ग्रन्थिः);
        पक्ष GGML_OP_GATED_DELTA_NET:
            सचे (ggml_webgpu_can_fuse_gdn_cache(cgraph, ग्रन्थिः_idx, अङ्कः_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[ग्रन्थिः_idx + अङ्कः_encoded_ops - 1]->src[1];
                पटिनिवत्त ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः->src[5],
                                                   ग्रन्थिः, dst_fuse);
            } अञ्ञथा {
                पटिनिवत्त ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ग्रन्थिः->src[3], ग्रन्थिः->src[4], ग्रन्थिः->src[5],
                                                   ग्रन्थिः, nullptr);
            }
        पक्ष GGML_OP_PAD:
            पटिनिवत्त ggml_webgpu_pad(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_ARGMAX:
            पटिनिवत्त ggml_webgpu_argmax(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_ARGSORT:
        पक्ष GGML_OP_शिखरम्_K:
            // we reuse the same argsort implementation for top_k
            पटिनिवत्त ggml_webgpu_argsort(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_CUMSUM:
            पटिनिवत्त ggml_webgpu_cumsum(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_SUM:
        पक्ष GGML_OP_SUM_ROWS:
            पटिनिवत्त ggml_webgpu_sum_rows(ctx, src0, ग्रन्थिः);
        पक्ष GGML_OP_CONV_2D:
            पटिनिवत्त ggml_webgpu_conv_2d(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_CONV_2D_DW:
            पटिनिवत्त ggml_webgpu_conv_2d_dw(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_IM2COL:
            पटिनिवत्त ggml_webgpu_im2col(ctx, src0, src1, ग्रन्थिः);
        पक्ष GGML_OP_UPSCALE:
            पटिनिवत्त ggml_webgpu_upscale(ctx, src0, ग्रन्थिः);
        स्वभाव:
            पटिनिवत्त std::nullopt;
    }
}

#ifdef GGML_WEBGPU_GPU_PROFILE
स्थिर शून्य ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ध्रुव std::vector<std::string> & pipeline_names,
                                                        uint32_t &                       अङ्कः_inflight_batches) {
    सचे (pipeline_names.empty()) {
        पटिनिवत्त;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.ResolveQuerySet(ctx->profile_timestamp_query_set, 0, ctx->profile_timestamp_query_गणनम्,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.CopyBufferToBuffer(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_गणनम् * परिमाण(uint64_t));

    wgpu::CommandBuffer profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, अङ्कः_inflight_batches);

    ध्रुव size_t mapped_size = ctx->profile_timestamp_query_गणनम् * परिमाण(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_गणनम् == 2 * pipeline_names.size());

    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,
                                   mapped_size);
    ध्रुव uint64_t * ts_data = (ध्रुव uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_size);

    अत्थाय (size_t i = 0; i < pipeline_names.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ध्रुव द्विगुण elapsed_ms = द्विगुण(ts_data[2 * i + 1] - ts_data[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#endif

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
स्थिर शून्य ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & अङ्कः_inflight_batches) {
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                               ctx->set_rows_host_error_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, अङ्कः_inflight_batches);
    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                   ctx->set_rows_host_error_buf.GetSize());
    ध्रुव uint32_t * error_data = (ध्रुव uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
    सचे (*error_data) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->set_rows_host_error_buf.Unmap();
#अञ्ञथा
    GGML_UNUSED(ctx);
    GGML_UNUSED(अङ्कः_inflight_batches);
#endif
}

स्थिर ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, संरचना ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_TOTAL_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t अङ्कः_batched_kernels  = 0;
    uint32_t अङ्कः_inflight_batches = 0;
    तार्किक     contains_set_rows    = मिच्छा;
    पूर्णाङ्क      अङ्कः_encoded_ops      = 1;
    पूर्णाङ्क      ग्रन्थिः_idx             = 0;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_गणनम् = 0;
    std::vector<std::string> profile_pipeline_names;
#endif

    ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    सचे (ctx->batch_compute_passes) {
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
    }

    याव (ग्रन्थिः_idx < cgraph->n_nodes) {
        सचे (cgraph->nodes[ग्रन्थिः_idx]->op == GGML_OP_SET_ROWS) {
            contains_set_rows = सच्च;
        }
        सचे (auto cmd = ggml_webgpu_encode(ctx, cgraph, ग्रन्थिः_idx, अङ्कः_encoded_ops)) {
            commands.push_back(*cmd);
            अङ्कः_batched_kernels += cmd.value().अङ्कः_kernels;
#ifdef GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_names.insert(profile_pipeline_names.end(), cmd->pipeline_names.begin(),
                                          cmd->pipeline_names.end());
#endif
        }

        सचे (अङ्कः_batched_kernels >= ctx->global_ctx->command_submit_batch_size) {
            सचे (ctx->active_compute_pass) {
                ctx->active_compute_pass.End();
            }
            अङ्कः_batched_kernels                = 0;
            wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, अङ्कः_inflight_batches);

            // reset state for next batch
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            सचे (ctx->batch_compute_passes) {
                ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#ifdef GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            सचे (ctx->profile_timestamp_query_गणनम् + 2 * ctx->global_ctx->command_submit_batch_size >=
                WEBGPU_MAX_PROFILE_QUERY_गणनम्) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, अङ्कः_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_गणनम् = 0;
                profile_pipeline_names.clear();
            }
#endif
        }

        ग्रन्थिः_idx += अङ्कः_encoded_ops;
        अङ्कः_encoded_ops = 1;
    }

    सचे (ctx->active_compute_pass) {
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
    }

    सचे (अङ्कः_batched_kernels > 0) {
        wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, अङ्कः_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->active_command_encoder = nullptr;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, अङ्कः_inflight_batches);
#endif

    सचे (contains_set_rows) {
        ggml_backend_webgpu_check_set_rows(ctx, अङ्कः_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);
    पटिनिवत्त GGML_STATUS_SUCCESS;
}

संरचना ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    तार्किक                  recorded = मिच्छा;
};

स्थिर ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    auto * event_ctx      = नूतन ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    auto * event   = नूतन ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    पटिनिवत्त event;
}

स्थिर शून्य ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete static_cast<ggml_backend_webgpu_event_context *>(event->context);
    delete event;
}

स्थिर शून्य ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    सचे (!event_ctx->recorded) {
        पटिनिवत्त;
    }
    wgpu::WaitStatus status =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    सचे (status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = मिच्छा;
}

स्थिर शून्य ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    event_ctx->recorded = सच्च;
}

स्थिर शून्य ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
}

स्थिर शून्य ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ध्रुव शून्य *   data,
                                                 size_t         offset,
                                                 size_t         size) {
    GGML_UNUSED(backend);
    auto * buf_ctx      = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // Write aligned portion
    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    सचे (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        अत्थाय (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((ध्रुव uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
}

स्थिर शून्य ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

स्थिर ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
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
    /* .event_record            = */ ggml_backend_webgpu_event_record,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ रिक्त,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

स्थिर शून्य ggml_backend_webgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_webgpu_buffer_context * ctx = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
    सचे (ctx != nullptr && ctx->buffer != nullptr) {
        ctx->buffer.Destroy();
        delete ctx;
    }
}

// Returns the "fake" base pointer.
स्थिर शून्य * ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    पटिनिवत्त webgpu_ptr_base;
}

स्थिर शून्य ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                offset,
                                                     size_t                size) {
    सचे (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        पटिनिवत्त;
    }

    WEBGPU_CPU_PROFILE_TOTAL_START(memset_tensor);

    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << offset << ", " << size << ")");

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32, total_offset, size);
    WEBGPU_CPU_PROFILE_TOTAL_END(memset_tensor, buf_ctx->global_ctx);
}

स्थिर शून्य ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  ध्रुव शून्य *          data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(set_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    सचे (total_offset % 4 != 0) {
        size_t lane = total_offset % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t head = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t head_val = 0;
        अत्थाय (size_t i = 0; i < head; i++) {
            ((uint8_t *) &head_val)[lane + i] = ((ध्रुव uint8_t *) data)[i];
        }
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, head_val, total_offset, head);

        total_offset += head;
        size -= head;
        data = (ध्रुव uint8_t *) data + head;
    }

    सचे (size > 0) {
        buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);
    }

    सचे (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        अत्थाय (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((ध्रुव uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
}

स्थिर शून्य ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  ध्रुव ggml_tensor *   tensor,
                                                  शून्य *                data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(get_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    size_t local_offset = total_offset % 4;
    सचे (local_offset != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        total_offset -= local_offset;
    }

    size_t final_size = size + local_offset;
    सचे (final_size % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_size += 4 - (final_size % 4);
    }

    std::lock_guard<std::recursive_mutex> lock(buf_ctx->global_ctx->mutex);

    सचे (buf_ctx->global_ctx->get_tensor_staging_buf == nullptr ||
        buf_ctx->global_ctx->get_tensor_staging_buf.GetSize() < final_size) {
        // Create a new staging buffer if it doesn't exist or is too small
        सचे (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_buffer(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_size,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(buf_ctx->buffer, total_offset, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_size);
    wgpu::CommandBuffer commands = encoder.Finish();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_buffer(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::MapMode::Read, 0, final_size);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ध्रुव शून्य * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_size);

    // Copy the data from the mapped range to the output buffer
    std::memcpy(data, (ध्रुव शून्य *) ((ध्रुव वर्ण *) mapped_range + local_offset), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, buf_ctx->global_ctx);
}

स्थिर शून्य ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_TOTAL_START(clear);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, value, 0, buffer->size);
    WEBGPU_CPU_PROFILE_TOTAL_END(clear, buf_ctx->global_ctx);
}

स्थिर ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_webgpu_buffer_get_base,
    /* .init_tensor     = */ रिक्त,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ रिक्त,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ रिक्त,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    पटिनिवत्त ctx->device_name.c_str();
}

स्थिर ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                          size_t                     size) {
    स्थिर std::atomic<पूर्णाङ्क> buffer_गणनम्;
    पूर्णाङ्क                     buffer_id = buffer_गणनम्++;
    std::string             buf_name  = "tensor_buf" + std::to_string(buffer_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << buffer_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::Buffer                         buf;
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_buffer_context * buf_ctx =
        नूतन ggml_backend_webgpu_buffer_context(buf, buf_name, ctx->webgpu_global_ctx);

    पटिनिवत्त ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);
}

स्थिर size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    पटिनिवत्त dev_ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
स्थिर size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    पटिनिवत्त dev_ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize;
}

स्थिर size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                             ध्रुव ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    विकल्प (tensor->op) {
        पक्ष GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            भङ्ग;
        पक्ष GGML_OP_शिखरम्_K:
            {
                ध्रुव ggml_tensor * src0 = tensor->src[0];
                सचे (src0) {
                    ध्रुव size_t full = परिमाण(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        full * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                ध्रुव ggml_tensor * Q            = tensor->src[0];
                ध्रुव ggml_tensor * K            = tensor->src[1];
                ध्रुव ggml_tensor * V            = tensor->src[2];
                ध्रुव ggml_tensor * mask         = tensor->src[3];
                ध्रुव auto &        capabilities = ctx->webgpu_global_ctx->capabilities;
                सचे (ggml_webgpu_flash_attn_use_vec_path(ctx->webgpu_global_ctx, Q, K, V)) {
                    ध्रुव तार्किक kv_direct =
                        ggml_webgpu_flash_attn_k_direct(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH) ||
                        ggml_webgpu_flash_attn_v_direct(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH);
                    ध्रुव uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.maxComputeWorkgroupStorageSize, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        mask != nullptr, kv_direct);

                    ध्रुव uint32_t vec_nwg_cap = capabilities.min_subgroup_size;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ध्रुव size_t   align = capabilities.limits.minStorageBufferOffsetAlignment;
                    ध्रुव uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    सचे (nwg > 1u) {
                        ध्रुव uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ध्रुव uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ध्रुव size_t   tmp_size_bytes = ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * परिमाण(भिन्न),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_size_bytes + align;
                    } अञ्ञथा {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    सचे (mask != nullptr) {
                        ध्रुव uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ध्रुव uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ध्रुव uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
                        ध्रुव uint32_t blk_batch_गणनम् = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ध्रुव uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_गणनम्;
                        ध्रुव size_t   blk_size_bytes =
                            ROUNDUP_POW2(blk_elems * परिमाण(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_size_bytes + align;
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
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_product,
                                             ctx->webgpu_global_ctx->vendor);
                सचे (use_mmvq) {
                    ध्रुव size_t q8_src1_size = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_size +
                                           ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                ध्रुव ggml_tensor * src0 = tensor->src[0];
                ध्रुव ggml_tensor * src1 = tensor->src[1];
                सचे (src0 && src1) {
                    ध्रुव size_t ids_cache_size = परिमाण(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    ध्रुव size_t ids_tiles_size = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_size + ids_tiles_size +
                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            भङ्ग;
        स्वभाव:
            भङ्ग;
    }
    पटिनिवत्त res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    पटिनिवत्त ctx->device_name.c_str();
}

स्थिर ध्रुव वर्ण * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    पटिनिवत्त ctx->device_desc.c_str();
}

स्थिर शून्य ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * मुक्त, size_t * total) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_buffer_size = ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#सचे UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_size = static_cast<uint64_t>(UINTPTR_MAX);
    सचे (max_buffer_size > max_ptr_size) {
        max_buffer_size = max_ptr_size;
    }
#endif
    *मुक्त  = static_cast<size_t>(max_buffer_size);
    *total = static_cast<size_t>(max_buffer_size);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    पटिनिवत्त GGML_BACKEND_DEVICE_TYPE_GPU;
}

स्थिर शून्य ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ मिच्छा,
        /* .host_buffer           = */ मिच्छा,
        /* .buffer_from_host_ptr  = */ मिच्छा,
        /* .events                = */ मिच्छा,
        /* .mmap_support          = */ सच्च,
    };
}

स्थिर ggml_guid_t ggml_backend_webgpu_guid(शून्य) {
    स्थिर ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    पटिनिवत्त &guid;
}

स्थिर शून्य ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_threads = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup *
                         ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_bytes_per_thread =
        CEIL_DIV(ctx->capabilities.limits.maxStorageBufferBindingSize, max_threads);
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].कुञ्चिका     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    constants[1].कुञ्चिका     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

स्थिर शून्य ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#सचे !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ध्रुव वर्ण * ध्रुव          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_गणनम् = 2;
    options.nextInChain                   = &adapterTogglesDesc;
#endif

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter _adapter, wgpu::StringView message) {
                सचे (status != wgpu::RequestAdapterStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(message).c_str());
                    पटिनिवत्त;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

स्थिर शून्य create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != nullptr);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#सचे !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    सचे (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.nextInChain = &subgroup_matrix_configs;
    }
#endif
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_size = ggml_backend_webgpu_get_command_submit_batch_size();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_product = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8IntegerDotProduct);

    तार्किक valid_subgroup_matrix_config = मिच्छा;
#सचे !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    सचे (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        अत्थाय (size_t i = 0; i < subgroup_matrix_configs.config_गणनम्; i++) {
            ध्रुव wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            सचे (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.resultComponentType == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = सच्च;
                भङ्ग;
            }
        }
    }
#endif
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_size = info.subgroupMinSize;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_size = info.subgroupMaxSize;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#सचे !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    सचे (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#endif

    सचे (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#endif

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.required_Feature_गणनम् = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](ध्रुव wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            सचे (reason == wgpu::DeviceLostReason::Destroyed) {
                पटिनिवत्त;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<पूर्णाङ्क>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](ध्रुव wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<पूर्णाङ्क>(reason),
                       std::string(message).c_str());
        });

#सचे !defined(__EMSCRIPTEN__) && !defined(__wasi__)
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

    dev_desc.nextInChain = &deviceTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                सचे (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                    पटिनिवत्त;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_SIZE_BYTES, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
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
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES,
        webgpu_ctx->global_ctx->command_submit_batch_size + WEBGPU_अङ्कः_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.minUniformBufferOffsetAlignment);
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_host_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "set_rows_host_error_buf");

#ifdef GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = मिच्छा;
    ggml_webgpu_create_buffer(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
        wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "profile_timestamp_host_buf");
    wgpu::QuerySetDescriptor query_set_desc = {};
    query_set_desc.type                     = wgpu::QueryType::Timestamp;
    query_set_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_गणनम्;
    webgpu_ctx->profile_timestamp_query_set = webgpu_ctx->global_ctx->device.CreateQuerySet(&query_set_desc);
#endif

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * परिमाण(uint32_t),
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "debug_host_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * परिमाण(uint32_t),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "debug_dev_buf");
#endif
    पटिनिवत्त webgpu_ctx;
}

स्थिर ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    auto * backend_ctx      = नूतन ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    auto * backend = नूतन ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    पटिनिवत्त backend;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    स्थिर संरचना ggml_backend_buffer_type ggml_backend_webgpu_buffer_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ रिक्त,
                        /* .get_alignment       = */ ggml_backend_webgpu_buffer_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_buffer_type_get_max_size,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ रिक्त,
                        /* .is_host             = */ रिक्त,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ रिक्त
    };

    पटिनिवत्त &ggml_backend_webgpu_buffer_type;
}

स्थिर तार्किक ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    पटिनिवत्त buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
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
            पटिनिवत्त सच्च;
        स्वभाव:
            पटिनिवत्त मिच्छा;
    }
}

स्थिर तार्किक ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    सचे (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        पटिनिवत्त मिच्छा;
    }

    तार्किक supports_op = मिच्छा;
    विकल्प (op->op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_RESHAPE:
            supports_op = सच्च;
            भङ्ग;
        पक्ष GGML_OP_ADD:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            भङ्ग;
        पक्ष GGML_OP_ADD_ID:
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
        पक्ष GGML_OP_SET:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            भङ्ग;
        पक्ष GGML_OP_SET_ROWS:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            भङ्ग;
        पक्ष GGML_OP_GET_ROWS:
            सचे (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } अञ्ञथा सचे (src0->type == GGML_TYPE_I32) {
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
                                supports_op = सच्च;
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
                            supports_op = सच्च;
                            भङ्ग;
                        स्वभाव:
                            भङ्ग;
                    }
                    भङ्ग;
                स्वभाव:
                    भङ्ग;
            }
            सचे (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = मिच्छा;
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
                सचे (!supports_op) {
                    भङ्ग;
                }
                सचे (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = मिच्छा;
                    भङ्ग;
                }
                ध्रुव auto & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ध्रुव size_t storage_offset_alignment = capabilities.limits.minStorageBufferOffsetAlignment;

                // subgroup matrix path requirements
                ध्रुव तार्किक use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
                    capabilities.supports_subgroup_matrix, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ध्रुव तार्किक float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_offset_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_offset_alignment));
                ध्रुव uint32_t k_tile_head_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_size(src1->type);
                ध्रुव uint32_t v_tile_head_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_size(src2->type);
                ध्रुव तार्किक     tile_kv_head_dims_aligned =
                    src0->ne[0] % k_tile_head_align == 0 && src2->ne[0] % v_tile_head_align == 0;
                ध्रुव तार्किक tile_can_dispatch_all_q_rows =
                    capabilities.limits.maxComputeInvocationsPerWorkgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_size;
                ध्रुव तार्किक use_tile = !use_subgroup_matrix && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_head_dims_aligned && tile_can_dispatch_all_q_rows;

                सचे (!use_subgroup_matrix && !use_tile) {
                    supports_op = मिच्छा;
                    भङ्ग;
                }
                ध्रुव uint32_t q_tile =
                    use_subgroup_matrix ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ध्रुव uint32_t kv_granularity = use_subgroup_matrix ? capabilities.sg_mat_n : 1u;
                ध्रुव तार्किक     kv_direct = use_subgroup_matrix ?
                                               ggml_webgpu_flash_attn_k_direct(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_direct(src0, src2, capabilities.sg_mat_k) :
                                               मिच्छा;
                ध्रुव uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.maxComputeWorkgroupStorageSize, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != nullptr, kv_direct);
                supports_op = max_kv_tile > 0;
                भङ्ग;
            }
        पक्ष GGML_OP_RMS_NORM:
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_rows(src0);
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
                पक्ष GGML_GLU_OP_GEGLU_QUICK:
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
                    पक्ष GGML_UNARY_OP_STEP:
                    पक्ष GGML_UNARY_OP_TANH:
                    पक्ष GGML_UNARY_OP_ELU:
                    पक्ष GGML_UNARY_OP_RELU:
                    पक्ष GGML_UNARY_OP_SIGMOID:
                    पक्ष GGML_UNARY_OP_GELU:
                    पक्ष GGML_UNARY_OP_GELU_QUICK:
                    पक्ष GGML_UNARY_OP_SILU:
                    पक्ष GGML_UNARY_OP_HARDSWISH:
                    पक्ष GGML_UNARY_OP_HARDSIGMOID:
                    पक्ष GGML_UNARY_OP_EXP:
                    पक्ष GGML_UNARY_OP_GELU_ERF:
                    पक्ष GGML_UNARY_OP_SOFTPLUS:
                    पक्ष GGML_UNARY_OP_EXPM1:
                    पक्ष GGML_UNARY_OP_FLOOR:
                    पक्ष GGML_UNARY_OP_CEIL:
                    पक्ष GGML_UNARY_OP_ROUND:
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
        पक्ष GGML_OP_SSM_SCAN:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            भङ्ग;
        पक्ष GGML_OP_GATED_DELTA_NET:
            {
                ध्रुव uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
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
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            भङ्ग;
        पक्ष GGML_OP_शिखरम्_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            भङ्ग;
        पक्ष GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            भङ्ग;
        पक्ष GGML_OP_SUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            भङ्ग;
        पक्ष GGML_OP_SUM_ROWS:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
            भङ्ग;
        पक्ष GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            भङ्ग;
        स्वभाव:
            भङ्ग;
    }
    सचे (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src2 != nullptr &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        supports_op = मिच्छा;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    सचे (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } अञ्ञथा {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    पटिनिवत्त supports_op;
}

स्थिर संरचना ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
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
    पटिनिवत्त ctx->name;
}

स्थिर size_t ggml_backend_webgpu_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    पटिनिवत्त ctx->device_गणनम्;
}

// Only one device is supported for now
स्थिर ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_TOTAL_START(reg_get_device);

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

    WEBGPU_CPU_PROFILE_TOTAL_END(reg_get_device, reg_ctx->webgpu_global_ctx);
    पटिनिवत्त &device;
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
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_गणनम् = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    सचे (ctx->webgpu_global_ctx != nullptr && ctx->webgpu_global_ctx->instance != nullptr) {
        पटिनिवत्त &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.data();
    instance_descriptor.required_Feature_गणनम्                 = instance_features.size();

#सचे !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ध्रुव वर्ण * ध्रुव          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_गणनम् = 1;
    instance_descriptor.nextInChain        = &instanceTogglesDesc;
#endif

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(नूतन webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    सचे (ctx->webgpu_global_ctx->instance != nullptr) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    सचे (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#सचे !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#endif
    ) {
        ctx->device_गणनम् = 1;
    }

    पटिनिवत्त &reg;
}

ggml_backend_t ggml_backend_webgpu_init(शून्य) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    सचे (ggml_backend_reg_dev_गणनम्(reg) == 0) {
        पटिनिवत्त nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    पटिनिवत्त ggml_backend_webgpu_backend_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
