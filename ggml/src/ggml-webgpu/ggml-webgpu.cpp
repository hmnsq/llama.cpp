/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#અન્તર્ભાવ "ggml-webgpu.h"

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-webgpu-shader-lib.hpp"
#અન્તર્ભાવ "ggml.h"

#ifdef __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#endif

#અન્તર્ભાવ <webgpu/webgpu_cpp.h>

#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstring>
#ifdef GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#endif
#જો defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#endif
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <optional>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <utility>
#અન્તર્ભાવ <vector>

#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
સ્થિર અંતઃરેખ શૂન્ય compute_2d_workgroups(uint32_t total_wg, uint32_t max_per_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(total_wg, max_per_dim));
    wg_x = CEIL_DIV(total_wg, wg_y);
}

સ્થિર અંતઃરેખ uint32_t ggml_webgpu_u32_from_f32(દશાંશ value) {
    uint32_t bits;
    memcpy(&bits, &value, કદ(bits));
    પાછા bits;
}

#ifdef GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(સંદેશ)  std::cout << સંદેશ << std::endl
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#નહીંતર
#    define WEBGPU_LOG_DEBUG(સંદેશ) ((શૂન્ય) 0)
#endif  // GGML_WEBGPU_DEBUG

#ifdef GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id) auto cpu_total_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)                                                         \
        auto   cpu_total_end_##id = std::chrono::high_resolution_clock::now();                            \
        દ્વિગુણ cpu_total_time_##id =                                                                      \
            std::chrono::duration<દ્વિગુણ, std::milli>(cpu_total_end_##id - cpu_total_start_##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_total_time_##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) auto cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)                                                          \
        auto   cpu_detail_end_##id = std::chrono::high_resolution_clock::now();                             \
        દ્વિગુણ cpu_detail_time_##id =                                                                       \
            std::chrono::duration<દ્વિગુણ, std::milli>(cpu_detail_end_##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#નહીંતર
#    define WEBGPU_CPU_PROFILE_TOTAL_START(id)
#    define WEBGPU_CPU_PROFILE_TOTAL_END(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)
#endif  // GGML_WEBGPU_CPU_PROFILE

#ifdef GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_ગણતરી        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES (WEBGPU_MAX_PROFILE_QUERY_ગણતરી * કદ(uint64_t))
#endif

/* Constants */

#define WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE 64u
#define WEBGPU_અંક_PARAM_SLOT_SAFETY_MARGIN      10u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#define WEBGPU_PARAMS_BUF_SIZE_BYTES             128  // enough for 32 parameters
#define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES     4
#define WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
સ્થિર શૂન્ય * અચલ webgpu_ptr_base = (શૂન્ય *) (uintptr_t) 0x1000;  // NOLINT

સ્થિર size_t ggml_webgpu_tensor_offset(અચલ ggml_tensor * tensor) {
    અચલ ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    પાછા (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
સ્થિર શૂન્ય ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      અચલ અક્ષર *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
સંરચના webgpu_param_arena {
    wgpu::Buffer buffer;
    size_t       slot_stride = 0;
    size_t       slot_size   = 0;
    uint32_t     slot_ગણતરી  = 0;
    uint32_t     next_slot   = 0;

    શૂન્ય init(wgpu::Device device, size_t slot_size, uint32_t slot_ગણતરી, size_t alignment) {
        આ->slot_stride = ROUNDUP_POW2(slot_size, alignment);
        આ->slot_size   = slot_size;
        આ->slot_ગણતરી  = slot_ગણતરી;
        આ->next_slot   = 0;

        ggml_webgpu_create_buffer(device, buffer, આ->slot_stride * slot_ગણતરી,
                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_slot(size_t size) {
        GGML_ASSERT(size <= slot_size);
        જો (next_slot >= slot_ગણતરી) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        પાછા slot_stride * next_slot++;
    }

    શૂન્ય reset() { next_slot = 0; }

    શૂન્ય cleanup() {
        જો (buffer) {
            buffer.Destroy();
            buffer = nullptr;
        }
    }

    ~webgpu_param_arena() { આ->cleanup(); }
};

સંરચના webgpu_encoded_op {
    uint32_t અંક_kernels = 0;
#ifdef GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_names;
#endif
};

સંરચના webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

સંરચના webgpu_capabilities {
    wgpu::Limits limits;
    તાર્કિક         supports_subgroups       = ખોટું;
    તાર્કિક         supports_subgroup_matrix = ખોટું;
    તાર્કિક         supports_dot_product     = ખોટું;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_size     = 0;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
    size_t   memset_bytes_per_thread;
};

// Stores global webgpu members
સંરચના webgpu_global_context_struct {
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
    std::unordered_map<std::string, દ્વિગુણ> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, દ્વિગુણ> cpu_detail_ms;
#endif

#ifdef GGML_WEBGPU_DEBUG
    wgpu::Buffer debug_host_buf;
    wgpu::Buffer debug_dev_buf;
#endif

    ~webgpu_global_context_struct() {
        જો (આ->get_tensor_staging_buf) {
            આ->get_tensor_staging_buf.Destroy();
            આ->get_tensor_staging_buf = nullptr;
        }
        જો (આ->memset_params_buf) {
            આ->memset_params_buf.Destroy();
            આ->memset_params_buf = nullptr;
        }
#ifdef GGML_WEBGPU_DEBUG
        જો (આ->debug_host_buf) {
            આ->debug_host_buf.Destroy();
            આ->debug_host_buf = nullptr;
        }
        જો (આ->debug_dev_buf) {
            આ->debug_dev_buf.Destroy();
            આ->debug_dev_buf = nullptr;
        }
#endif
    }
};

પ્રકારવ્યાખ્યા std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
સંરચના webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::Buffer             set_rows_dev_error_buf;
    wgpu::Buffer             set_rows_host_error_buf;
    wgpu::CommandEncoder     active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;
    તાર્કિક                     batch_compute_passes = સાચું;

    size_t memset_bytes_per_thread;

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, દ્વિગુણ> shader_gpu_time_ms;
    wgpu::Buffer                            profile_timestamp_dev_buf;
    wgpu::Buffer                            profile_timestamp_host_buf;
    wgpu::QuerySet                          profile_timestamp_query_set;
    uint32_t                                profile_timestamp_query_ગણતરી = 0;
#endif

    ~webgpu_context_struct() {
#ifdef GGML_WEBGPU_GPU_PROFILE
        જો (આ->profile_timestamp_host_buf) {
            આ->profile_timestamp_host_buf.Destroy();
            આ->profile_timestamp_host_buf = nullptr;
        }
        જો (આ->profile_timestamp_dev_buf) {
            આ->profile_timestamp_dev_buf.Destroy();
            આ->profile_timestamp_dev_buf = nullptr;
        }
        જો (આ->profile_timestamp_query_set) {
            આ->profile_timestamp_query_set.Destroy();
            આ->profile_timestamp_query_set = nullptr;
        }
#endif
        જો (આ->set_rows_host_error_buf) {
            આ->set_rows_host_error_buf.Destroy();
            આ->set_rows_host_error_buf = nullptr;
        }
        જો (આ->set_rows_dev_error_buf) {
            આ->set_rows_dev_error_buf.Destroy();
            આ->set_rows_dev_error_buf = nullptr;
        }
    }
};

પ્રકારવ્યાખ્યા std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
સંરચના ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_ગણતરી;
    અચલ અક્ષર *          name;
};

// Per-device struct for the global logical device interface
સંરચના ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
સંરચના ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
સંરચના ggml_backend_webgpu_buffer_context {
    wgpu::Buffer          buffer;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_buffer_context(wgpu::Buffer buf, std::string lbl, webgpu_global_context global_ctx_) :
        buffer(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

સ્થિર webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   અચલ અક્ષર *                             shader_code,
                                                   અચલ અક્ષર *                             label,
                                                   અચલ std::vector<wgpu::ConstantEntry> & constants = {}) {
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
    જો (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.data();
        pipeline_desc.compute.constant_ગણતરી = constants.size();
    }
    પાછા { device.CreateComputePipeline(&pipeline_desc), label };
}

સ્થિર શૂન્ય ggml_webgpu_create_buffer(wgpu::Device &    device,
                                      wgpu::Buffer &    buffer,
                                      size_t            size,
                                      wgpu::BufferUsage usage,
                                      અચલ અક્ષર *      label) {
    wgpu::BufferDescriptor buffer_desc;
    buffer_desc.size             = size;
    buffer_desc.usage            = usage;
    buffer_desc.label            = label;
    buffer_desc.mappedAtCreation = ખોટું;

    // TODO: error handling
    buffer = device.CreateBuffer(&buffer_desc);
}

સ્થિર wgpu::Buffer ggml_webgpu_tensor_buf(અચલ ggml_tensor * tensor) {
    ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    પાછા ctx->buffer;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
સ્થિર size_t ggml_webgpu_tensor_align_offset(અચલ ggml_tensor * t, size_t alignment) {
    અચલ size_t offset    = ggml_webgpu_tensor_offset(t);
    અચલ size_t type_size = ggml_type_size(t->type);
    size_t       aligned   = offset & ~(alignment - 1);
    જ્યારે ((offset - aligned) % type_size != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    પાછા aligned;
}

સ્થિર size_t ggml_webgpu_tensor_misalignment(અચલ ggml_tensor * t, size_t alignment) {
    પાછા ggml_webgpu_tensor_offset(t) - ggml_webgpu_tensor_align_offset(t, alignment);
}

સ્થિર size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, અચલ ggml_tensor * t) {
    પાછા ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

સ્થિર size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, અચલ ggml_tensor * t) {
    પાછા ggml_webgpu_tensor_align_offset(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

સ્થિર size_t ggml_webgpu_tensor_binding_size(અચલ ggml_tensor * t, size_t alignment) {
    પાછા ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

સ્થિર size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, અચલ ggml_tensor * t) {
    પાછા ggml_webgpu_tensor_binding_size(t, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
}

સ્થિર તાર્કિક ggml_webgpu_tensor_binding_overlap(અચલ webgpu_global_context & global_ctx,
                                               અચલ ggml_tensor *           a,
                                               અચલ ggml_tensor *           b) {
    જો (a->buffer != b->buffer) {
        પાછા ખોટું;
    }

    અચલ size_t alignment = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    અચલ size_t a_offset  = ggml_webgpu_tensor_align_offset(a, alignment);
    અચલ size_t b_offset  = ggml_webgpu_tensor_align_offset(b, alignment);
    પાછા a_offset < b_offset + ggml_webgpu_tensor_binding_size(b, alignment) &&
           b_offset < a_offset + ggml_webgpu_tensor_binding_size(a, alignment);
}

સ્થિર તાર્કિક ggml_webgpu_tensor_binding_overlap_range(અચલ webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_buffer_t         buffer,
                                                     size_t                        offset,
                                                     size_t                        size) {
    જો (tensor->buffer != buffer) {
        પાછા ખોટું;
    }

    અચલ size_t alignment     = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    અચલ size_t tensor_offset = ggml_webgpu_tensor_align_offset(tensor, alignment);
    પાછા tensor_offset < offset + size && offset < tensor_offset + ggml_webgpu_tensor_binding_size(tensor, alignment);
}

સંરચના ggml_webgpu_merged_binding_range {
    size_t offset;
    size_t size;
};

સ્થિર ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_offset = SIZE_MAX;
    size_t merged_end    = 0;

    માટે (ggml_tensor * tensor : tensors) {
        અચલ size_t bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
        અચલ size_t bind_end    = bind_offset + ggml_webgpu_tensor_binding_size(ctx, tensor);

        merged_offset = std::min(merged_offset, bind_offset);
        merged_end    = std::max(merged_end, bind_end);
    }

    પાછા { merged_offset, merged_end - merged_offset };
}

સ્થિર uint32_t ggml_webgpu_tensor_merged_element_offset(અચલ ggml_tensor *                      tensor,
                                                         અચલ ggml_webgpu_merged_binding_range & merged_range) {
    પાછા (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_size(tensor->type));
}

સ્થિર wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
                                                              wgpu::Buffer buffer,
                                                              uint64_t     offset,
                                                              uint64_t     size) {
    wgpu::BindGroupEntry entry = {};
    entry.binding              = binding;
    entry.buffer               = std::move(buffer);
    entry.offset               = offset;
    entry.size                 = size;
    પાછા entry;
}

સ્થિર wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    પાછા ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_offset(ctx, tensor),
                                             ggml_webgpu_tensor_binding_size(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

template <typename T>
સ્થિર શૂન્ય ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T                callback_status,
                                                  T                success_status,
                                                  અચલ અક્ષર *     wait_name,
                                                  અચલ અક્ષર *     failure_name,
                                                  અચલ અક્ષર *     callback_message) {
    જો (wait_status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    જો (wait_status == wgpu::WaitStatus::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    જો (callback_status != success_status) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<પૂર્ણાંક>(callback_status),
                   callback_message);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
સ્થિર uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    પાછા UINT32_MAX;
}

સ્થિર uint32_t ggml_backend_webgpu_get_command_submit_batch_size() {
    પાછા WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
}

સ્થિર શૂન્ય ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::QueueWorkDoneStatus callback_status = wgpu::QueueWorkDoneStatus::Error;
    std::string               callback_message;

    અચલ wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
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

સ્થિર શૂન્ય ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
                                           wgpu::Buffer &          buffer,
                                           wgpu::MapMode           mode,
                                           size_t                  offset,
                                           size_t                  size) {
    wgpu::MapAsyncStatus callback_status = wgpu::MapAsyncStatus::Error;
    std::string          callback_message;

    અચલ wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
                        [&callback_status, &callback_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                            callback_status  = status;
                            callback_message = std::string(message);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::MapAsyncStatus::Success,
                                          "Buffer map wait", "Buffer map", callback_message.c_str());
}

સ્થિર શૂન્ય ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                અચલ wgpu::CommandBuffer commands,
                                                uint32_t &                અંક_inflight_batches) {
    જો (અંક_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        અંક_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    અંક_inflight_batches++;
}

#ifdef GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
સ્થિર શૂન્ય ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());
    અચલ દશાંશ * debug_data = (અચલ દશાંશ *) ctx->debug_host_buf.GetConstMappedRange();
    std::cout << "debug[0]: " << debug_data[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#endif

સ્થિર webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         અચલ std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            result = {};
    std::vector<wgpu::BindGroup> bind_groups;
    std::vector<size_t>          param_offsets;
    result.અંક_kernels = dispatches.size();

    માટે (size_t i = 0; i < dispatches.size(); i++) {
        અચલ webgpu_dispatch_desc & dispatch     = dispatches[i];
        અચલ size_t                 param_size   = dispatch.params.size() * કદ(uint32_t);
        અચલ size_t                 param_offset = ctx->param_arena.alloc_slot(param_size);

        std::vector<wgpu::BindGroupEntry> entries            = dispatch.bind_group_entries;
        uint32_t                          params_binding_અંક = entries.size();
        entries.push_back(ggml_webgpu_make_bind_group_entry(params_binding_અંક, ctx->param_arena.buffer, param_offset,
                                                            ctx->param_arena.slot_size));

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = dispatch.pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entry_ગણતરી = entries.size();
        bind_group_desc.entries    = entries.data();
        bind_group_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.CreateBindGroup(&bind_group_desc));
        param_offsets.push_back(param_offset);
    }

    માટે (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.WriteBuffer(ctx->param_arena.buffer, param_offsets[i], dispatches[i].params.data(),
                                           dispatches[i].params.size() * કદ(uint32_t));
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    માટે (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_ગણતરી + 2 <= WEBGPU_MAX_PROFILE_QUERY_ગણતરી);
        અચલ uint32_t query_begin = ctx->profile_timestamp_query_ગણતરી++;
        અચલ uint32_t query_end   = ctx->profile_timestamp_query_ગણતરી++;

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
#નહીંતર
    માટે (size_t i = 0; i < dispatches.size(); i++) {
        જો (ctx->batch_compute_passes) {
            ctx->active_compute_pass.SetPipeline(dispatches[i].pipeline.pipeline);
            ctx->active_compute_pass.SetBindGroup(0, bind_groups[i]);
            ctx->active_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second,
                                                        1);
        } નહીંતર {
            wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass();
            pass.SetPipeline(dispatches[i].pipeline.pipeline);
            pass.SetBindGroup(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
            pass.End();
        }
    }
#endif

    પાછા result;
}

સ્થિર webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    પાછા ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_group_entries), { wg_x, wg_y } },
    });
}

સ્થિર શૂન્ય ggml_backend_webgpu_buffer_memset(webgpu_global_context & ctx,
                                              wgpu::Buffer &          buf,
                                              uint32_t                value,
                                              size_t                  offset,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) offset, (uint32_t) size, value };
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_bind_group_entry(0, buf, 0, buf.GetSize()) };
    size_t                            bytes_per_wg =
        ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup * ctx->capabilities.memset_bytes_per_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, bytes_per_wg);

    ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * કદ(uint32_t));

    wgpu::BindGroupEntry params_entry = {};
    params_entry.binding              = 1;
    params_entry.buffer               = ctx->memset_params_buf;
    params_entry.offset               = 0;
    params_entry.size                 = WEBGPU_PARAMS_BUF_SIZE_BYTES;
    entries.push_back(params_entry);

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
    bind_group_desc.entry_ગણતરી = entries.size();
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

સ્થિર અચલ અક્ષર * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    પાછા ctx->name.c_str();
}

સ્થિર શૂન્ય ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ifdef GGML_WEBGPU_CPU_PROFILE
    std::cout << "\n[ggml_webgpu cpu profiling summary]\n";
    દ્વિગુણ total_cpu = 0.0;
    માટે (અચલ auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        total_cpu += kv.second;
    }
    std::cout << "ggml_webgpu: total cpu time: " << total_cpu << " ms\n";
    std::cout << "ggml_webgpu: cpu breakdown:\n";
    માટે (અચલ auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        દ્વિગુણ pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    જો (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::cout << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    માટે (અચલ auto & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        દ્વિગુણ pct = (total_cpu > 0.0) ? (kv.second / total_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    std::cout << "\n[ggml_webgpu gpu profiling summary]\n";
    દ્વિગુણ total_gpu = 0.0;
    માટે (અચલ auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        total_gpu += kv.second;
    }
    std::cout << "ggml_webgpu: total gpu time (all shaders): " << total_gpu << " ms\n";
    std::cout << "\nggml_webgpu: gpu breakdown:\n";
    માટે (અચલ auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        દ્વિગુણ pct = (total_gpu > 0.0) ? (kv.second / total_gpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#endif

#જો defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::cout << "ggml_webgpu: gpu/cpu ratio: " << (total_cpu > 0.0 ? total_gpu / total_cpu : 0.0) << "\n";
#endif

    delete ctx;
    delete backend;
}

સ્થિર webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
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
    અચલ તાર્કિક inplace   = decisions->inplace;

    અચલ uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    અચલ uint32_t dst_type_size = (uint32_t) ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
        (uint32_t) (((અચલ int32_t *) dst->op_params)[3] / dst_type_size),

        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),

        1u,
        (uint32_t) (((અચલ int32_t *) dst->op_params)[0] / dst_type_size),
        (uint32_t) (((અચલ int32_t *) dst->op_params)[1] / dst_type_size),
        (uint32_t) (((અચલ int32_t *) dst->op_params)[2] / dst_type_size),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          binding_index = 0;
    જો (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        binding_index++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    અચલ uint32_t ne = (uint32_t) ggml_nelements(dst);

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
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
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

    અચલ uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_size);
    અચલ uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    અચલ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    અચલ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    અચલ int32_t p0 = ggml_get_op_params_i32(dst, 2);
    અચલ int32_t p1 = ggml_get_op_params_i32(dst, 3);
    અચલ int32_t d0 = ggml_get_op_params_i32(dst, 4);
    અચલ int32_t d1 = ggml_get_op_params_i32(dst, 5);

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

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
સ્થિર webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    અચલ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    અચલ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    અચલ int32_t p0 = ggml_get_op_params_i32(dst, 2);
    અચલ int32_t p1 = ggml_get_op_params_i32(dst, 3);
    અચલ int32_t d0 = ggml_get_op_params_i32(dst, 4);
    અચલ int32_t d1 = ggml_get_op_params_i32(dst, 5);

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
    અચલ તાર્કિક      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    અચલ int32_t s0    = ggml_get_op_params_i32(dst, 0);
    અચલ int32_t s1    = ggml_get_op_params_i32(dst, 1);
    અચલ int32_t p0    = ggml_get_op_params_i32(dst, 2);
    અચલ int32_t p1    = ggml_get_op_params_i32(dst, 3);
    અચલ int32_t d0    = ggml_get_op_params_i32(dst, 4);
    અચલ int32_t d1    = ggml_get_op_params_i32(dst, 5);
    અચલ તાર્કિક    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    અચલ uint32_t KW = src0->ne[0];
    અચલ uint32_t KH = is_2D ? src0->ne[1] : 1;
    અચલ uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    અચલ uint32_t IW = src1->ne[0];
    અચલ uint32_t IH = is_2D ? src1->ne[1] : 1;
    અચલ uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    અચલ uint32_t OW = dst->ne[1];
    અચલ uint32_t OH = is_2D ? dst->ne[2] : 1;

    અચલ uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_size(src1->type));
    અચલ uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0;
    અચલ uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_size(src1->type));
    અચલ uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_size(src1->type));

    અચલ uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_size(dst->type));
    અચલ uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_size(dst->type));
    અચલ uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)) : 0;
    અચલ uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)) :
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

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
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

    અચલ uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_per_wg);

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

    અચલ uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_size);
    અચલ uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
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
    જો (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    જો (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    જો (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
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

    જો (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
    } નહીંતર જો (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    } નહીંતર જો (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_offset = 0;
    size_t xbc_bind_size   = 0;
    જો (xbc_overlap > 0) {
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

    uint32_t binding_અંક = 0;

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, src0),
    };
    // xbc_merged binding
    જો (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_અંક++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_offset, xbc_bind_size));
    }
    // x
    જો (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, src1));
    }
    // B
    જો (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, src4));
    }
    // C
    જો (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક++, dst));

    અચલ uint32_t total_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    અચલ uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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

    અચલ uint32_t s_v      = (uint32_t) src2->ne[0];
    અચલ uint32_t h        = (uint32_t) src2->ne[1];
    અચલ uint32_t n_tokens = (uint32_t) src2->ne[2];
    અચલ uint32_t n_seqs   = (uint32_t) src2->ne[3];
    અચલ uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    અચલ દશાંશ    scale    = 1.0f / sqrtf((દશાંશ) s_v);
    uint32_t       scale_u32;
    memcpy(&scale_u32, &scale, કદ(scale_u32));

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

    જો (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst_fuse));
    }

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

સ્થિર std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    જો (ggml_is_empty(src) || ggml_is_empty(idx)) {
        પાછા std::nullopt;
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

    જો (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->set_rows_dev_error_buf, 0,
                                                            ctx->set_rows_dev_error_buf.GetSize()));
    }

    uint32_t threads;
    જો (ggml_is_quantized(dst->type)) {
        અચલ uint32_t blocks_per_row = src->ne[0] / ggml_blck_size(dst->type);
        threads =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_blocks ? (blocks_per_row / 2) : blocks_per_row);
    } નહીંતર જો (decisions->vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } નહીંતર {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(threads, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
સ્થિર std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_size_entry(uint32_t wg_size) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].ચાવી   = "wg_size";
    constants[0].value = wg_size;
    પાછા constants;
}

સ્થિર webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    અચલ તાર્કિક float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    અચલ uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    અચલ uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    અચલ uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    અચલ uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    અચલ uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));

    અચલ તાર્કિક vec4_aligned = offset_src % 4 == 0 && offset_dst % 4 == 0 && stride_src1 % 4 == 0 &&
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
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર શૂન્ય ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
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
    અચલ size_t dst_offset           = ggml_webgpu_tensor_offset(dst);
    અચલ size_t q8_src1_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    અચલ size_t q8_src1_binding_size = ROUNDUP_POW2(
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
    અચલ uint32_t wg_per_vec     = (src0->ne[0] / 4 + (q8_wg_size - 1)) / q8_wg_size;
    અચલ uint32_t q8_total_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_per_vec;
    અચલ uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(q8_total_wg, max_wg_per_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

સ્થિર webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    તાર્કિક use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    તાર્કિક use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_product,
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
    અચલ તાર્કિક src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    જો (use_mat_vec) {
        જો (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } નહીંતર {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    જો (src_overlap) {
        અચલ ggml_webgpu_merged_binding_range merged_range =
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
    જો (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        auto & mmvq_qq8_entry = dispatches[0].bind_group_entries[1];
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_entry.offset,
                                                            mmvq_qq8_entry.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    } નહીંતર જો (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } નહીંતર {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    અચલ uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    જો (use_mat_vec) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
        uint32_t total_wg      = output_groups * batches;
        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
    } નહીંતર {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        જો (decisions->use_subgroup_matrix) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_matrix_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_matrix_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } નહીંતર {
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

    પાછા ggml_backend_webgpu_build_multi(ctx, dispatches);
}

સ્થિર webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    અચલ uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    અચલ uint32_t param_n_expert_used = (uint32_t) dst->ne[1];

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

    અચલ uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
    uint32_t       total_wg       = output_groups * param_n_expert_used;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
સ્થિર uint32_t ggml_webgpu_mul_mat_id_max_tiles(અચલ ggml_tensor * dst) {
    અચલ uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    અચલ uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    પાછા CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
સ્થિર size_t ggml_webgpu_mul_mat_id_tiles_nbytes(અચલ ggml_tensor * dst) {
    પાછા (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * કદ(uint32_t);
}

સ્થિર webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    જો (dst->ne[2] == 1) {
        પાછા ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
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

    અચલ uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    અચલ uint32_t param_n_expert_used = (uint32_t) dst->ne[1];
    અચલ uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
    };

    અચલ size_t dst_offset       = ggml_webgpu_tensor_offset(dst);
    અચલ size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * કદ(uint32_t);
    અચલ size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    અચલ size_t ids_cache_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    અચલ size_t ids_tiles_align_offset =
        ROUNDUP_POW2(ids_cache_align_offset + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);

    અચલ size_t ids_cache_bind_size = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    અચલ size_t ids_tiles_bind_size = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

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

    અચલ uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_size_m;
    અચલ uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    અચલ uint32_t total_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    પાછા ggml_backend_webgpu_build_multi(ctx, dispatches);
}

સંરચના ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> entries;
    size_t                            kv_bind_offset = 0;
    size_t                            kv_bind_size   = 0;
    તાર્કિક                              has_mask       = ખોટું;
    તાર્કિક                              has_sinks      = ખોટું;
    તાર્કિક                              kv_overlap     = ખોટું;
};

સ્થિર તાર્કિક ggml_webgpu_flash_attn_use_vec_path(અચલ webgpu_global_context & global_ctx,
                                                અચલ ggml_tensor *           Q,
                                                અચલ ggml_tensor *           K,
                                                અચલ ggml_tensor *           V) {
    અચલ size_t storage_offset_alignment = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;

    અચલ તાર્કિક k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment);
    અચલ તાર્કિક v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);

    અચલ uint32_t k_vec_head_align =
        ggml_is_quantized(K->type) ? ggml_blck_size(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    અચલ uint32_t v_vec_head_align =
        ggml_is_quantized(V->type) ? ggml_blck_size(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    અચલ તાર્કિક kv_vec_head_dims_aligned = Q->ne[0] % k_vec_head_align == 0 && V->ne[0] % v_vec_head_align == 0;

    પાછા global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_head_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

સ્થિર ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    mask,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    દશાંશ scale         = ggml_get_op_params_f32(dst, 0);
    દશાંશ max_bias      = ggml_get_op_params_f32(dst, 1);
    દશાંશ logit_softcap = ggml_get_op_params_f32(dst, 2);
    જો (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    દશાંશ n_head_log2 = દશાંશ(1u << (uint32_t) floor(log2(Q->ne[2])));
    દશાંશ m0          = powf(2.0f, -(max_bias) / n_head_log2);
    દશાંશ m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

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
    જો (op.kv_overlap) {
        અચલ ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
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
    જો (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_size));
    } નહીંતર {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
    }
    uint32_t binding_index = op.kv_overlap ? 2u : 3u;
    જો (op.has_mask) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, mask));
    }
    જો (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, dst));

    પાછા op;
}

સ્થિર uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    અચલ uint64_t kv_span = (uint64_t) kv_tile;
    જ્યારે ((2u * nwg * kv_span) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    પાછા std::min(nwg, vec_nwg_cap);
}

સ્થિર webgpu_encoded_op ggml_webgpu_flash_attn_direct(webgpu_context & ctx, અચલ ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_per_head = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_per_head * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    પાછા ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

સ્થિર webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
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
    uint32_t     blk_batch_ગણતરી = 0;

    અચલ uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_size;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    અચલ uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    અચલ તાર્કિક     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_base  = 0;
    uint64_t     tmp_size_bytes  = 0;
    wgpu::Buffer tmp_buf         = {};
    uint64_t     tmp_bind_offset = 0;
    uint64_t     tmp_bind_size   = 0;
    અચલ size_t align_bytes     = ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
    અચલ size_t dst_offset      = ggml_webgpu_tensor_offset(dst);
    size_t       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);

    જો (use_vec_reduce) {
        અચલ uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        અચલ uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_base                 = tmp_data_elems;
        tmp_size_bytes =
            ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * કદ(દશાંશ), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_size   = tmp_size_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
    } નહીંતર {
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
    જો (op.has_mask) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        અચલ uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
        blk_batch_ગણતરી             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        અચલ uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_ગણતરી;
        blk_size_bytes              = ROUNDUP_POW2(blk_elems * કદ(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        અચલ ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
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
    જો (op.has_mask) {
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
    જો (op.kv_overlap) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_size));
    } નહીંતર {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
                                                                  ggml_webgpu_tensor_binding_size(ctx, K)));
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
                                                                  ggml_webgpu_tensor_binding_size(ctx, V)));
    }
    uint32_t split_binding_index = op.kv_overlap ? 2u : 3u;
    જો (op.has_mask) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(mask),
                                                                  ggml_webgpu_tensor_align_offset(ctx, mask),
                                                                  ggml_webgpu_tensor_binding_size(ctx, mask)));
    }
    જો (op.has_sinks) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_offset(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_size(ctx, sinks)));
    }
    જો (op.has_mask) {
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
    જો (use_vec_reduce) {
        અચલ uint32_t reduce_sg_size = ctx->global_ctx->capabilities.max_subgroup_size;
        અચલ uint32_t reduce_wg_size = std::max(
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
    અચલ uint64_t split_wg_total = (uint64_t) wg_x * nwg;
    GGML_ASSERT(split_wg_total <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    જો (op.has_mask) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_ગણતરી }
        });
    }
    dispatches.push_back({
        pipeline, std::move(split_params), std::move(split_entries), { (uint32_t) split_wg_total, 1u }
    });
    જો (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    પાછા ggml_backend_webgpu_build_multi(ctx, dispatches);
}

સ્થિર webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    mask,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, mask, sinks, dst);
    જો (ggml_webgpu_flash_attn_use_vec_path(ctx->global_ctx, Q, K, V)) {
        પાછા ggml_webgpu_flash_attn_vec(ctx, Q, K, V, mask, sinks, dst, std::move(op));
    }
    પાછા ggml_webgpu_flash_attn_direct(ctx, op);
}

સ્થિર webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    તાર્કિક is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    અચલ તાર્કિક inplace   = decisions->inplace;

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
    જો (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        પસંદગી (unary_op) {
            સ્થિતિ GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    દશાંશ alpha_n = ggml_get_op_params_f32(dst, 1);
                    દશાંશ alpha_p = ggml_get_op_params_f32(dst, 2);
                    દશાંશ beta    = ggml_get_op_params_f32(dst, 3);
                    દશાંશ eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    તોડો;
                }
            મૂળભૂત:
                તોડો;
        }
    } નહીંતર જો (dst->op == GGML_OP_CLAMP) {
        દશાંશ clamp_min = ggml_get_op_params_f32(dst, 0);
        દશાંશ clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } નહીંતર જો (dst->op == GGML_OP_FILL) {
        દશાંશ fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, effective_src),
    };
    જો (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_દ્વિઅંકી_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    અચલ તાર્કિક      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_દ્વિઅંકી_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_દ્વિઅંકી_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
    size_t src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    જો (decisions->src_overlap) {
        અચલ ggml_webgpu_merged_binding_range merged_range =
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

    જો (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } નહીંતર {
        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        જો (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_add_id(webgpu_context & ctx,
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

    જો (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       total_wg       = ggml_nrows(dst);
    અચલ uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    જો (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        પાછા {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    અચલ તાર્કિક      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_દ્વિઅંકી_shader_decisions *>(pipeline.context.get());

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_size   = 0;
    જો (decisions->src_overlap) {
        જો (ggml_nbytes(src0) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src1);
            merged_size   = ggml_webgpu_tensor_binding_size(ctx, src1);
            offset_src0   = 0;
            offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
        } નહીંતર જો (ggml_nbytes(src1) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
            merged_size   = ggml_webgpu_tensor_binding_size(ctx, src0);
            offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
            offset_src1   = 0;
        } નહીંતર {
            અચલ ggml_webgpu_merged_binding_range merged_range =
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
    જો (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } નહીંતર {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ne, decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
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
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    જો (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } નહીંતર જો (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } નહીંતર {
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

    અચલ તાર્કિક      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    જો (decisions->src_overlap) {
        અચલ ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_offset  = merged_range.offset;
        merged_size    = merged_range.size;
        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
        params[0]      = offset_rn_src;
        params[1]      = offset_mul_src;
    }

    જો (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
    } નહીંતર જો (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } નહીંતર {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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
    જો (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
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

    અચલ તાર્કિક inplace         = decisions->inplace;
    અચલ પૂર્ણાંક  has_freq_factor = (src2 != nullptr);

    અચલ પૂર્ણાંક n_dims     = ((int32_t *) dst->op_params)[1];
    અચલ પૂર્ણાંક mode       = ((int32_t *) dst->op_params)[2];
    અચલ પૂર્ણાંક n_ctx_orig = ((int32_t *) dst->op_params)[4];
    અચલ પૂર્ણાંક n_offs     = ((int32_t *) dst->op_params)[15];

    દશાંશ freq_base;
    દશાંશ freq_scale;
    દશાંશ ext_factor;
    દશાંશ attn_factor;
    દશાંશ beta_fast;
    દશાંશ beta_slow;
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, કદ(દશાંશ));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, કદ(દશાંશ));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, કદ(દશાંશ));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, કદ(દશાંશ));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, કદ(દશાંશ));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, કદ(દશાંશ));

    પૂર્ણાંક sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * કદ(પૂર્ણાંક));

    દશાંશ theta_scale = powf(freq_base, -2.0f / n_dims);

    દશાંશ corr_dims[2];
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
    જો (has_freq_factor) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
    }
    જો (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    અચલ તાર્કિક      src_overlap = src1 != nullptr && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    auto * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    અચલ પૂર્ણાંક split = (src1 != nullptr);

    uint32_t offset_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
    uint32_t offset_src1 =
        src1 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)) : 0;
    size_t merged_offset = 0;
    size_t merged_size   = 0;
    જો (decisions->src_overlap) {
        અચલ ggml_webgpu_merged_binding_range merged_range =
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
    જો (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
    } નહીંતર {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    }
    જો (split && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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

    જો (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
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

    અચલ તાર્કિક inplace     = decisions->inplace;
    અચલ પૂર્ણાંક  has_mask    = (src1 != nullptr);
    અચલ પૂર્ણાંક  has_sink    = (src2 != nullptr);
    દશાંશ      max_bias    = ggml_get_op_params_f32(dst, 1);
    દશાંશ      n_head_log2 = દશાંશ(1u << (uint32_t) floor(log2(src0->ne[2])));
    દશાંશ      m0          = powf(2.0f, -(max_bias) / n_head_log2);
    દશાંશ      m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

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
    uint32_t                          binding_અંક = 1;
    જો (has_mask) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_અંક, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_offset(ctx, src1),
                                                            ggml_webgpu_tensor_binding_size(ctx, src1)));
        binding_અંક++;
    }
    જો (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક, src2));
        binding_અંક++;
    }
    જો (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_અંક, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t total_wg = ggml_nrows(dst);
    compute_2d_workgroups(total_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

સ્થિર webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    તાર્કિક is_ટોચ_k = dst->op == GGML_OP_ટોચ_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    auto * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    અચલ uint32_t src_ne0 = (uint32_t) src->ne[0];
    અચલ uint32_t nrows   = (uint32_t) ggml_nrows(src);
    અચલ uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_size);
    અચલ uint32_t block_size =
        is_ટોચ_k ? std::min(argsort_decisions->wg_size, (uint32_t) dst->ne[0]) : argsort_decisions->wg_size;
    uint32_t out_ne0 = src_ne0;
    જો (is_ટોચ_k) {
        જો (npr > 1) {
            અચલ uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_size;
            out_ne0                  = (npr - 1) * block_size + std::min(last_tile, block_size);
        } નહીંતર {
            out_ne0 = block_size;
        }
    }

    uint32_t merge_len    = block_size;
    uint32_t merge_passes = 0;
    જ્યારે (merge_len < out_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    અચલ તાર્કિક start_in_tmp = (merge_passes % 2) == 1;

    અચલ size_t dst_offset = ggml_webgpu_tensor_offset(dst);
    અચલ size_t idx_nbytes = out_ne0 * ggml_nrows(dst) * કદ(int32_t);
    અચલ size_t tmp_offset =
        ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
    અચલ size_t tmp_binding_size = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    અચલ size_t dst_binding_size =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    અચલ uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type));
    અચલ uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type));
    અચલ uint32_t offset_tmp  = 0;
    અચલ uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_size(src->type));
    અચલ uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_size(src->type));
    અચલ uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_size(src->type));
    અચલ uint32_t stride_idx1 = out_ne0;
    અચલ uint32_t stride_idx2 = out_ne0 * (uint32_t) dst->ne[1];
    અચલ uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    અચલ uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    અચલ size_t   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    અચલ size_t   init_binding_size = start_in_tmp ? tmp_binding_size : dst_binding_size;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_size,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    અચલ uint32_t total_wg_init  = npr * nrows;
    અચલ uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(total_wg_init, max_wg_per_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::BindGroupEntry> init_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), init_align_offset, init_binding_size)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    જો (merge_passes == 0) {
        પાછા ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    તાર્કિક     in_is_tmp = start_in_tmp;
    uint32_t len       = block_size;
    જ્યારે (len < out_ne0) {
        અચલ uint32_t nm = CEIL_DIV(out_ne0, 2 * len);

        અચલ તાર્કિક     out_is_tmp  = !in_is_tmp;
        અચલ uint32_t offset_in   = in_is_tmp ? offset_tmp : offset_dst;
        અચલ uint32_t offset_out  = out_is_tmp ? offset_tmp : offset_dst;
        અચલ size_t   align_in    = in_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        અચલ size_t   align_out   = out_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        અચલ size_t   size_in     = in_is_tmp ? tmp_binding_size : dst_binding_size;
        અચલ size_t   size_out    = out_is_tmp ? tmp_binding_size : dst_binding_size;
        અચલ uint32_t ટોચ_k_out   = (is_ટોચ_k && nm == 1) ? (uint32_t) dst->ne[0] : out_ne0;
        અચલ uint32_t stride_out1 = ટોચ_k_out;
        અચલ uint32_t stride_out2 = ટોચ_k_out * (uint32_t) dst->ne[1];
        અચલ uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

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
                                               ટોચ_k_out,
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
        અચલ uint32_t total_wg_merge = nm * nrows;
        compute_2d_workgroups(total_wg_merge, max_wg_per_dim, wg_x_merge, wg_y_merge);

        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    પાછા ggml_backend_webgpu_build_multi(ctx, dispatches);
}

સ્થિર webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    તાર્કિક                  total_sum = dst->op == GGML_OP_SUM;
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
    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

સ્થિર તાર્કિક ggml_webgpu_can_fuse_rms_norm_mul(અચલ સંરચના ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx) {
    જો (!ggml_can_fuse(cgraph, ગાંઠ_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        પાછા ખોટું;
    }

    // additional constraints specific to this fusion
    અચલ ggml_tensor * rms_norm = cgraph->nodes[ગાંઠ_idx];
    અચલ ggml_tensor * mul      = cgraph->nodes[ગાંઠ_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    જો (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    જો (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        પાછા ખોટું;
    }
    // rms_norm shader assumes contiguous rows
    જો (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_webgpu_can_fuse_gdn_cache(અચલ સંરચના ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx, પૂર્ણાંક & અંક_encoded_ops) {
    અચલ ggml_tensor * gdn = cgraph->nodes[ગાંઠ_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    જો (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        પાછા ખોટું;
    }

    અચલ ggml_tensor * src_v     = gdn->src[2];
    અચલ int64_t       S_v       = src_v->ne[0];
    અચલ int64_t       H         = src_v->ne[1];
    અચલ int64_t       n_tokens  = src_v->ne[2];
    અચલ int64_t       n_seqs    = src_v->ne[3];
    અચલ int64_t       D         = S_v * S_v * H;
    અચલ int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    અચલ int64_t       n_written = std::min<int64_t>(n_tokens, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    અચલ size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    અચલ ggml_tensor * cpy     = nullptr;
    પૂર્ણાંક                 cpy_idx = 0;
    માટે (પૂર્ણાંક j = ગાંઠ_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        અચલ ggml_tensor * n = cgraph->nodes[j];
        જો (ggml_op_is_empty(n->op) || ggml_is_empty(n)) {
            ચાલુ;
        }
        જો (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            પાછા ખોટું;
        }
        cpy     = n;
        cpy_idx = j;
    }
    જો (cpy == nullptr) {
        પાછા ખોટું;
    }

    અચલ ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    અચલ ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    જો (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != tail_off ||
        !ggml_is_contiguous(cpy_src)) {
        પાછા ખોટું;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    અચલ std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    જો (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        પાછા ખોટું;
    }

    અંક_encoded_ops = cpy_idx - ગાંઠ_idx + 1;

    પાછા સાચું;
}

સ્થિર webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    અચલ uint32_t        mode_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
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

    પાછા ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
સ્થિર std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           પૂર્ણાંક            ગાંઠ_idx,
                                                           પૂર્ણાંક &          અંક_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  ગાંઠ  = nodes[ગાંઠ_idx];

    જો (ggml_is_empty(ગાંઠ)) {
        પાછા std::nullopt;
    }
    જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        પાછા std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << ગાંઠ << ", " << ggml_op_name(ગાંઠ->op) << ")");

    ggml_tensor * src0 = ગાંઠ->src[0];
    ggml_tensor * src1 = ગાંઠ->src[1];
    ggml_tensor * src2 = ગાંઠ->src[2];

    પસંદગી (ગાંઠ->op) {
            // no-ops
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
        સ્થિતિ GGML_OP_RESHAPE:
            પાછા std::nullopt;
        સ્થિતિ GGML_OP_DUP:
        સ્થિતિ GGML_OP_CPY:
        સ્થિતિ GGML_OP_CONT:
            પાછા ggml_webgpu_cpy(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_SET:
            પાછા ggml_webgpu_set(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_SET_ROWS:
            પાછા ggml_webgpu_set_rows(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_GET_ROWS:
            પાછા ggml_webgpu_get_rows(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_MUL_MAT:
            પાછા ggml_webgpu_mul_mat(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            પાછા ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, ગાંઠ);
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            પાછા ggml_webgpu_flash_attn(ctx, src0, src1, src2, ગાંઠ->src[3], ગાંઠ->src[4], ગાંઠ);
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_SUB:
        સ્થિતિ GGML_OP_MUL:
        સ્થિતિ GGML_OP_DIV:
            પાછા ggml_webgpu_દ્વિઅંકી_op(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_ADD_ID:
            પાછા ggml_webgpu_add_id(ctx, src0, src1, src2, ગાંઠ);
        સ્થિતિ GGML_OP_CONCAT:
            પાછા ggml_webgpu_concat(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_REPEAT:
            પાછા ggml_webgpu_repeat(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_RMS_NORM:
            જો (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, ગાંઠ_idx)) {
                અંક_encoded_ops        = 2;
                ggml_tensor * mul_ગાંઠ = nodes[ગાંઠ_idx + 1];
                પાછા ggml_webgpu_rms_norm_mul(ctx, src0, ગાંઠ, mul_ગાંઠ->src[0], mul_ગાંઠ->src[1], mul_ગાંઠ);
            } નહીંતર {
                પાછા ggml_webgpu_row_norm(ctx, src0, ગાંઠ);
            }
        સ્થિતિ GGML_OP_NORM:
        સ્થિતિ GGML_OP_L2_NORM:
            પાછા ggml_webgpu_row_norm(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_ROPE:
            પાછા ggml_webgpu_rope(ctx, src0, src1, src2, ગાંઠ);
        સ્થિતિ GGML_OP_GLU:
            પાછા ggml_webgpu_glu(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_SCALE:
            પાછા ggml_webgpu_scale(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_SOFT_MAX:
            પાછા ggml_webgpu_soft_max(ctx, src0, src1, src2, ગાંઠ);
        સ્થિતિ GGML_OP_UNARY:
        સ્થિતિ GGML_OP_CLAMP:
        સ્થિતિ GGML_OP_FILL:
        સ્થિતિ GGML_OP_LOG:
        સ્થિતિ GGML_OP_SQR:
        સ્થિતિ GGML_OP_SQRT:
        સ્થિતિ GGML_OP_SIN:
        સ્થિતિ GGML_OP_COS:
        સ્થિતિ GGML_OP_DIAG:
        સ્થિતિ GGML_OP_TRI:
            પાછા ggml_webgpu_unary_op(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_SOLVE_TRI:
            પાછા ggml_webgpu_solve_tri(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_SSM_CONV:
            પાછા ggml_webgpu_ssm_conv(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_SSM_SCAN:
            પાછા ggml_webgpu_ssm_scan(ctx, src0, src1, src2, ગાંઠ->src[3], ગાંઠ->src[4], ગાંઠ->src[5], ગાંઠ->src[6],
                                        ગાંઠ);
        સ્થિતિ GGML_OP_GATED_DELTA_NET:
            જો (ggml_webgpu_can_fuse_gdn_cache(cgraph, ગાંઠ_idx, અંક_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[ગાંઠ_idx + અંક_encoded_ops - 1]->src[1];
                પાછા ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ગાંઠ->src[3], ગાંઠ->src[4], ગાંઠ->src[5],
                                                   ગાંઠ, dst_fuse);
            } નહીંતર {
                પાછા ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, ગાંઠ->src[3], ગાંઠ->src[4], ગાંઠ->src[5],
                                                   ગાંઠ, nullptr);
            }
        સ્થિતિ GGML_OP_PAD:
            પાછા ggml_webgpu_pad(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_ARGMAX:
            પાછા ggml_webgpu_argmax(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_ARGSORT:
        સ્થિતિ GGML_OP_ટોચ_K:
            // we reuse the same argsort implementation for top_k
            પાછા ggml_webgpu_argsort(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_CUMSUM:
            પાછા ggml_webgpu_cumsum(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_SUM:
        સ્થિતિ GGML_OP_SUM_ROWS:
            પાછા ggml_webgpu_sum_rows(ctx, src0, ગાંઠ);
        સ્થિતિ GGML_OP_CONV_2D:
            પાછા ggml_webgpu_conv_2d(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_CONV_2D_DW:
            પાછા ggml_webgpu_conv_2d_dw(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_IM2COL:
            પાછા ggml_webgpu_im2col(ctx, src0, src1, ગાંઠ);
        સ્થિતિ GGML_OP_UPSCALE:
            પાછા ggml_webgpu_upscale(ctx, src0, ગાંઠ);
        મૂળભૂત:
            પાછા std::nullopt;
    }
}

#ifdef GGML_WEBGPU_GPU_PROFILE
સ્થિર શૂન્ય ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        અચલ std::vector<std::string> & pipeline_names,
                                                        uint32_t &                       અંક_inflight_batches) {
    જો (pipeline_names.empty()) {
        પાછા;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.ResolveQuerySet(ctx->profile_timestamp_query_set, 0, ctx->profile_timestamp_query_ગણતરી,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.CopyBufferToBuffer(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_ગણતરી * કદ(uint64_t));

    wgpu::CommandBuffer profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, અંક_inflight_batches);

    અચલ size_t mapped_size = ctx->profile_timestamp_query_ગણતરી * કદ(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_ગણતરી == 2 * pipeline_names.size());

    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,
                                   mapped_size);
    અચલ uint64_t * ts_data = (અચલ uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_size);

    માટે (size_t i = 0; i < pipeline_names.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        અચલ દ્વિગુણ elapsed_ms = દ્વિગુણ(ts_data[2 * i + 1] - ts_data[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#endif

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
સ્થિર શૂન્ય ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & અંક_inflight_batches) {
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                               ctx->set_rows_host_error_buf.GetSize());
    wgpu::CommandBuffer commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, અંક_inflight_batches);
    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                   ctx->set_rows_host_error_buf.GetSize());
    અચલ uint32_t * error_data = (અચલ uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
    જો (*error_data) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->set_rows_host_error_buf.Unmap();
#નહીંતર
    GGML_UNUSED(ctx);
    GGML_UNUSED(અંક_inflight_batches);
#endif
}

સ્થિર ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, સંરચના ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_TOTAL_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t અંક_batched_kernels  = 0;
    uint32_t અંક_inflight_batches = 0;
    તાર્કિક     contains_set_rows    = ખોટું;
    પૂર્ણાંક      અંક_encoded_ops      = 1;
    પૂર્ણાંક      ગાંઠ_idx             = 0;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_ગણતરી = 0;
    std::vector<std::string> profile_pipeline_names;
#endif

    ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    જો (ctx->batch_compute_passes) {
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
    }

    જ્યારે (ગાંઠ_idx < cgraph->n_nodes) {
        જો (cgraph->nodes[ગાંઠ_idx]->op == GGML_OP_SET_ROWS) {
            contains_set_rows = સાચું;
        }
        જો (auto cmd = ggml_webgpu_encode(ctx, cgraph, ગાંઠ_idx, અંક_encoded_ops)) {
            commands.push_back(*cmd);
            અંક_batched_kernels += cmd.value().અંક_kernels;
#ifdef GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_names.insert(profile_pipeline_names.end(), cmd->pipeline_names.begin(),
                                          cmd->pipeline_names.end());
#endif
        }

        જો (અંક_batched_kernels >= ctx->global_ctx->command_submit_batch_size) {
            જો (ctx->active_compute_pass) {
                ctx->active_compute_pass.End();
            }
            અંક_batched_kernels                = 0;
            wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, અંક_inflight_batches);

            // reset state for next batch
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            જો (ctx->batch_compute_passes) {
                ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#ifdef GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            જો (ctx->profile_timestamp_query_ગણતરી + 2 * ctx->global_ctx->command_submit_batch_size >=
                WEBGPU_MAX_PROFILE_QUERY_ગણતરી) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, અંક_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_ગણતરી = 0;
                profile_pipeline_names.clear();
            }
#endif
        }

        ગાંઠ_idx += અંક_encoded_ops;
        અંક_encoded_ops = 1;
    }

    જો (ctx->active_compute_pass) {
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
    }

    જો (અંક_batched_kernels > 0) {
        wgpu::CommandBuffer batch_commands = ctx->active_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, અંક_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->active_command_encoder = nullptr;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, અંક_inflight_batches);
#endif

    જો (contains_set_rows) {
        ggml_backend_webgpu_check_set_rows(ctx, અંક_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);
    પાછા GGML_STATUS_SUCCESS;
}

સંરચના ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    તાર્કિક                  recorded = ખોટું;
};

સ્થિર ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    auto * event_ctx      = નવું ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    auto * event   = નવું ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    પાછા event;
}

સ્થિર શૂન્ય ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete static_cast<ggml_backend_webgpu_event_context *>(event->context);
    delete event;
}

સ્થિર શૂન્ય ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    જો (!event_ctx->recorded) {
        પાછા;
    }
    wgpu::WaitStatus status =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    જો (status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = ખોટું;
}

સ્થિર શૂન્ય ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    event_ctx->recorded = સાચું;
}

સ્થિર શૂન્ય ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
}

સ્થિર શૂન્ય ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 અચલ શૂન્ય *   data,
                                                 size_t         offset,
                                                 size_t         size) {
    GGML_UNUSED(backend);
    auto * buf_ctx      = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // Write aligned portion
    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);

    જો (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        માટે (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((અચલ uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
}

સ્થિર શૂન્ય ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

સ્થિર ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
    /* .get_tensor_async        = */ ખાલી,
    /* .set_tensor_2d_async     = */ ખાલી,
    /* .get_tensor_2d_async     = */ ખાલી,
    /* .cpy_tensor_async        = */ ખાલી,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_record,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ ખાલી,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

સ્થિર શૂન્ય ggml_backend_webgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_webgpu_buffer_context * ctx = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
    જો (ctx != nullptr && ctx->buffer != nullptr) {
        ctx->buffer.Destroy();
        delete ctx;
    }
}

// Returns the "fake" base pointer.
સ્થિર શૂન્ય * ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    પાછા webgpu_ptr_base;
}

સ્થિર શૂન્ય ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                offset,
                                                     size_t                size) {
    જો (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        પાછા;
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

સ્થિર શૂન્ય ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  અચલ શૂન્ય *          data,
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
    જો (total_offset % 4 != 0) {
        size_t lane = total_offset % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t head = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t head_val = 0;
        માટે (size_t i = 0; i < head; i++) {
            ((uint8_t *) &head_val)[lane + i] = ((અચલ uint8_t *) data)[i];
        }
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, head_val, total_offset, head);

        total_offset += head;
        size -= head;
        data = (અચલ uint8_t *) data + head;
    }

    જો (size > 0) {
        buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);
    }

    જો (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_size = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        માટે (size_t i = 0; i < remaining_size; i++) {
            ((uint8_t *) &val32)[i] = ((અચલ uint8_t *) data)[size - remaining_size + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
                                          total_offset + (size - remaining_size), remaining_size);
    }
    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
}

સ્થિર શૂન્ય ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  અચલ ggml_tensor *   tensor,
                                                  શૂન્ય *                data,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_TOTAL_START(get_tensor);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                              << ", " << offset << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    size_t local_offset = total_offset % 4;
    જો (local_offset != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        total_offset -= local_offset;
    }

    size_t final_size = size + local_offset;
    જો (final_size % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_size += 4 - (final_size % 4);
    }

    std::lock_guard<std::recursive_mutex> lock(buf_ctx->global_ctx->mutex);

    જો (buf_ctx->global_ctx->get_tensor_staging_buf == nullptr ||
        buf_ctx->global_ctx->get_tensor_staging_buf.GetSize() < final_size) {
        // Create a new staging buffer if it doesn't exist or is too small
        જો (buf_ctx->global_ctx->get_tensor_staging_buf) {
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
    અચલ શૂન્ય * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_size);

    // Copy the data from the mapped range to the output buffer
    std::memcpy(data, (અચલ શૂન્ય *) ((અચલ અક્ષર *) mapped_range + local_offset), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, buf_ctx->global_ctx);
}

સ્થિર શૂન્ય ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_TOTAL_START(clear);
    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
    ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, value, 0, buffer->size);
    WEBGPU_CPU_PROFILE_TOTAL_END(clear, buf_ctx->global_ctx);
}

સ્થિર ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_webgpu_buffer_get_base,
    /* .init_tensor     = */ ખાલી,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ખાલી,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ ખાલી,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

સ્થિર અચલ અક્ષર * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    પાછા ctx->device_name.c_str();
}

સ્થિર ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                          size_t                     size) {
    સ્થિર std::atomic<પૂર્ણાંક> buffer_ગણતરી;
    પૂર્ણાંક                     buffer_id = buffer_ગણતરી++;
    std::string             buf_name  = "tensor_buf" + std::to_string(buffer_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << buffer_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::Buffer                         buf;
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_buffer_context * buf_ctx =
        નવું ggml_backend_webgpu_buffer_context(buf, buf_name, ctx->webgpu_global_ctx);

    પાછા ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);
}

સ્થિર size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    પાછા dev_ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
સ્થિર size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    પાછા dev_ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize;
}

સ્થિર size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                             અચલ ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    પસંદગી (tensor->op) {
        સ્થિતિ GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            તોડો;
        સ્થિતિ GGML_OP_ટોચ_K:
            {
                અચલ ggml_tensor * src0 = tensor->src[0];
                જો (src0) {
                    અચલ size_t full = કદ(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        full * 2 + ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            {
                અચલ ggml_tensor * Q            = tensor->src[0];
                અચલ ggml_tensor * K            = tensor->src[1];
                અચલ ggml_tensor * V            = tensor->src[2];
                અચલ ggml_tensor * mask         = tensor->src[3];
                અચલ auto &        capabilities = ctx->webgpu_global_ctx->capabilities;
                જો (ggml_webgpu_flash_attn_use_vec_path(ctx->webgpu_global_ctx, Q, K, V)) {
                    અચલ તાર્કિક kv_direct =
                        ggml_webgpu_flash_attn_k_direct(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH) ||
                        ggml_webgpu_flash_attn_v_direct(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH);
                    અચલ uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.maxComputeWorkgroupStorageSize, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        mask != nullptr, kv_direct);

                    અચલ uint32_t vec_nwg_cap = capabilities.min_subgroup_size;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    અચલ size_t   align = capabilities.limits.minStorageBufferOffsetAlignment;
                    અચલ uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    જો (nwg > 1u) {
                        અચલ uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        અચલ uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        અચલ size_t   tmp_size_bytes = ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * કદ(દશાંશ),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_size_bytes + align;
                    } નહીંતર {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    જો (mask != nullptr) {
                        અચલ uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        અચલ uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        અચલ uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
                        અચલ uint32_t blk_batch_ગણતરી = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        અચલ uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_ગણતરી;
                        અચલ size_t   blk_size_bytes =
                            ROUNDUP_POW2(blk_elems * કદ(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_size_bytes + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT:
            {
                અચલ ggml_tensor * src0 = tensor->src[0];
                અચલ ggml_tensor * src1 = tensor->src[1];
                તાર્કિક                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_product,
                                             ctx->webgpu_global_ctx->vendor);
                જો (use_mmvq) {
                    અચલ size_t q8_src1_size = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_size +
                                           ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            {
                અચલ ggml_tensor * src0 = tensor->src[0];
                અચલ ggml_tensor * src1 = tensor->src[1];
                જો (src0 && src1) {
                    અચલ size_t ids_cache_size = કદ(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    અચલ size_t ids_tiles_size = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_size + ids_tiles_size +
                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            તોડો;
        મૂળભૂત:
            તોડો;
    }
    પાછા res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

સ્થિર અચલ અક્ષર * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    પાછા ctx->device_name.c_str();
}

સ્થિર અચલ અક્ષર * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    પાછા ctx->device_desc.c_str();
}

સ્થિર શૂન્ય ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_buffer_size = ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#જો UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_size = static_cast<uint64_t>(UINTPTR_MAX);
    જો (max_buffer_size > max_ptr_size) {
        max_buffer_size = max_ptr_size;
    }
#endif
    *મુક્ત  = static_cast<size_t>(max_buffer_size);
    *total = static_cast<size_t>(max_buffer_size);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    પાછા GGML_BACKEND_DEVICE_TYPE_GPU;
}

સ્થિર શૂન્ય ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, સંરચના ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ખોટું,
        /* .host_buffer           = */ ખોટું,
        /* .buffer_from_host_ptr  = */ ખોટું,
        /* .events                = */ ખોટું,
        /* .mmap_support          = */ સાચું,
    };
}

સ્થિર ggml_guid_t ggml_backend_webgpu_guid(શૂન્ય) {
    સ્થિર ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    પાછા &guid;
}

સ્થિર શૂન્ય ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_threads = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup *
                         ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_bytes_per_thread =
        CEIL_DIV(ctx->capabilities.limits.maxStorageBufferBindingSize, max_threads);
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].ચાવી     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    constants[1].ચાવી     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

સ્થિર શૂન્ય ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#જો !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    અચલ અક્ષર * અચલ          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_ગણતરી = 2;
    options.nextInChain                   = &adapterTogglesDesc;
#endif

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter _adapter, wgpu::StringView message) {
                જો (status != wgpu::RequestAdapterStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(message).c_str());
                    પાછા;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

સ્થિર શૂન્ય create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != nullptr);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#જો !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    જો (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
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

    તાર્કિક valid_subgroup_matrix_config = ખોટું;
#જો !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    જો (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        માટે (size_t i = 0; i < subgroup_matrix_configs.config_ગણતરી; i++) {
            અચલ wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            જો (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.resultComponentType == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = સાચું;
                તોડો;
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

#જો !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    જો (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#endif

    જો (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#endif

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.required_Feature_ગણતરી = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](અચલ wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            જો (reason == wgpu::DeviceLostReason::Destroyed) {
                પાછા;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<પૂર્ણાંક>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](અચલ wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<પૂર્ણાંક>(reason),
                       std::string(message).c_str());
        });

#જો !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    અચલ અક્ષર * અચલ          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    અચલ અક્ષર * અચલ          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_ગણતરી  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_ગણતરી = 1;

    dev_desc.nextInChain = &deviceTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                જો (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                    પાછા;
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

સ્થિર webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES,
        webgpu_ctx->global_ctx->command_submit_batch_size + WEBGPU_અંક_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.minUniformBufferOffsetAlignment);
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_host_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "set_rows_host_error_buf");

#ifdef GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = ખોટું;
    ggml_webgpu_create_buffer(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
        wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_SIZE_BYTES,
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "profile_timestamp_host_buf");
    wgpu::QuerySetDescriptor query_set_desc = {};
    query_set_desc.type                     = wgpu::QueryType::Timestamp;
    query_set_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_ગણતરી;
    webgpu_ctx->profile_timestamp_query_set = webgpu_ctx->global_ctx->device.CreateQuerySet(&query_set_desc);
#endif

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * કદ(uint32_t),
                              wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, "debug_host_buf");
    ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * કદ(uint32_t),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "debug_dev_buf");
#endif
    પાછા webgpu_ctx;
}

સ્થિર ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    auto * backend_ctx      = નવું ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    auto * backend = નવું ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    પાછા backend;
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_webgpu_buffer_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ ખાલી,
                        /* .get_alignment       = */ ggml_backend_webgpu_buffer_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_buffer_type_get_max_size,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ ખાલી,
                        /* .is_host             = */ ખાલી,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ ખાલી
    };

    પાછા &ggml_backend_webgpu_buffer_type;
}

સ્થિર તાર્કિક ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    પાછા buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
}

સ્થિર તાર્કિક ggml_webgpu_supported_qtype(ggml_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ2_S:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_NVFP4:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

સ્થિર તાર્કિક ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    જો (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        પાછા ખોટું;
    }

    તાર્કિક supports_op = ખોટું;
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
        સ્થિતિ GGML_OP_RESHAPE:
            supports_op = સાચું;
            તોડો;
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_SUB:
        સ્થિતિ GGML_OP_MUL:
        સ્થિતિ GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_ADD_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            તોડો;
        સ્થિતિ GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            તોડો;
        સ્થિતિ GGML_OP_DUP:
        સ્થિતિ GGML_OP_CPY:
        સ્થિતિ GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            તોડો;
        સ્થિતિ GGML_OP_SET:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            તોડો;
        સ્થિતિ GGML_OP_SET_ROWS:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            તોડો;
        સ્થિતિ GGML_OP_GET_ROWS:
            જો (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } નહીંતર જો (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT:
            {
                પસંદગી (src1->type) {
                    સ્થિતિ GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        તોડો;
                    સ્થિતિ GGML_TYPE_F32:
                        પસંદગી (src0->type) {
                            સ્થિતિ GGML_TYPE_F32:
                            સ્થિતિ GGML_TYPE_F16:
                            સ્થિતિ GGML_TYPE_BF16:
                            સ્થિતિ GGML_TYPE_Q1_0:
                            સ્થિતિ GGML_TYPE_Q2_0:
                            સ્થિતિ GGML_TYPE_Q4_0:
                            સ્થિતિ GGML_TYPE_Q4_1:
                            સ્થિતિ GGML_TYPE_Q5_0:
                            સ્થિતિ GGML_TYPE_Q5_1:
                            સ્થિતિ GGML_TYPE_Q8_0:
                            સ્થિતિ GGML_TYPE_Q2_K:
                            સ્થિતિ GGML_TYPE_Q3_K:
                            સ્થિતિ GGML_TYPE_Q4_K:
                            સ્થિતિ GGML_TYPE_Q5_K:
                            સ્થિતિ GGML_TYPE_Q6_K:
                            સ્થિતિ GGML_TYPE_IQ2_XXS:
                            સ્થિતિ GGML_TYPE_IQ2_XS:
                            સ્થિતિ GGML_TYPE_IQ2_S:
                            સ્થિતિ GGML_TYPE_IQ3_XXS:
                            સ્થિતિ GGML_TYPE_IQ3_S:
                            સ્થિતિ GGML_TYPE_IQ1_S:
                            સ્થિતિ GGML_TYPE_IQ1_M:
                            સ્થિતિ GGML_TYPE_IQ4_NL:
                            સ્થિતિ GGML_TYPE_IQ4_XS:
                            સ્થિતિ GGML_TYPE_MXFP4:
                            સ્થિતિ GGML_TYPE_NVFP4:
                                supports_op = સાચું;
                                તોડો;
                            મૂળભૂત:
                                તોડો;
                        }
                    મૂળભૂત:
                        તોડો;
                }
                તોડો;
            }
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            પસંદગી (src1->type) {
                સ્થિતિ GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    તોડો;
                સ્થિતિ GGML_TYPE_F32:
                    પસંદગી (src0->type) {
                        સ્થિતિ GGML_TYPE_F32:
                        સ્થિતિ GGML_TYPE_F16:
                        સ્થિતિ GGML_TYPE_BF16:
                        સ્થિતિ GGML_TYPE_Q1_0:
                        સ્થિતિ GGML_TYPE_Q4_0:
                        સ્થિતિ GGML_TYPE_Q4_1:
                        સ્થિતિ GGML_TYPE_Q5_0:
                        સ્થિતિ GGML_TYPE_Q5_1:
                        સ્થિતિ GGML_TYPE_Q8_0:
                        સ્થિતિ GGML_TYPE_Q2_K:
                        સ્થિતિ GGML_TYPE_Q3_K:
                        સ્થિતિ GGML_TYPE_Q4_K:
                        સ્થિતિ GGML_TYPE_Q5_K:
                        સ્થિતિ GGML_TYPE_Q6_K:
                        સ્થિતિ GGML_TYPE_IQ1_S:
                        સ્થિતિ GGML_TYPE_IQ1_M:
                        સ્થિતિ GGML_TYPE_IQ2_XXS:
                        સ્થિતિ GGML_TYPE_IQ2_XS:
                        સ્થિતિ GGML_TYPE_IQ2_S:
                        સ્થિતિ GGML_TYPE_IQ3_XXS:
                        સ્થિતિ GGML_TYPE_IQ3_S:
                        સ્થિતિ GGML_TYPE_IQ4_NL:
                        સ્થિતિ GGML_TYPE_IQ4_XS:
                        સ્થિતિ GGML_TYPE_MXFP4:
                        સ્થિતિ GGML_TYPE_NVFP4:
                            supports_op = સાચું;
                            તોડો;
                        મૂળભૂત:
                            તોડો;
                    }
                    તોડો;
                મૂળભૂત:
                    તોડો;
            }
            જો (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = ખોટું;
            }
            તોડો;
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                જો (!supports_op) {
                    તોડો;
                }
                જો (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = ખોટું;
                    તોડો;
                }
                અચલ auto & capabilities             = ctx->webgpu_global_ctx->capabilities;
                અચલ size_t storage_offset_alignment = capabilities.limits.minStorageBufferOffsetAlignment;

                // subgroup matrix path requirements
                અચલ તાર્કિક use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
                    capabilities.supports_subgroup_matrix, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                અચલ તાર્કિક float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_offset_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_offset_alignment));
                અચલ uint32_t k_tile_head_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_size(src1->type);
                અચલ uint32_t v_tile_head_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_size(src2->type);
                અચલ તાર્કિક     tile_kv_head_dims_aligned =
                    src0->ne[0] % k_tile_head_align == 0 && src2->ne[0] % v_tile_head_align == 0;
                અચલ તાર્કિક tile_can_dispatch_all_q_rows =
                    capabilities.limits.maxComputeInvocationsPerWorkgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_size;
                અચલ તાર્કિક use_tile = !use_subgroup_matrix && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_head_dims_aligned && tile_can_dispatch_all_q_rows;

                જો (!use_subgroup_matrix && !use_tile) {
                    supports_op = ખોટું;
                    તોડો;
                }
                અચલ uint32_t q_tile =
                    use_subgroup_matrix ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                અચલ uint32_t kv_granularity = use_subgroup_matrix ? capabilities.sg_mat_n : 1u;
                અચલ તાર્કિક     kv_direct = use_subgroup_matrix ?
                                               ggml_webgpu_flash_attn_k_direct(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_direct(src0, src2, capabilities.sg_mat_k) :
                                               ખોટું;
                અચલ uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.maxComputeWorkgroupStorageSize, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != nullptr, kv_direct);
                supports_op = max_kv_tile > 0;
                તોડો;
            }
        સ્થિતિ GGML_OP_RMS_NORM:
        સ્થિતિ GGML_OP_NORM:
        સ્થિતિ GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_rows(src0);
            તોડો;
        સ્થિતિ GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            તોડો;
        સ્થિતિ GGML_OP_GLU:
            પસંદગી (ggml_get_glu_op(op)) {
                સ્થિતિ GGML_GLU_OP_REGLU:
                સ્થિતિ GGML_GLU_OP_GEGLU:
                સ્થિતિ GGML_GLU_OP_SWIGLU:
                સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
                સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
                સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    તોડો;
                સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    તોડો;
                મૂળભૂત:
                    તોડો;
            }
            તોડો;
        સ્થિતિ GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_UNARY:
            {
                અચલ ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                પસંદગી (UNARY_OP) {
                    સ્થિતિ GGML_UNARY_OP_ABS:
                    સ્થિતિ GGML_UNARY_OP_SGN:
                    સ્થિતિ GGML_UNARY_OP_NEG:
                    સ્થિતિ GGML_UNARY_OP_STEP:
                    સ્થિતિ GGML_UNARY_OP_TANH:
                    સ્થિતિ GGML_UNARY_OP_ELU:
                    સ્થિતિ GGML_UNARY_OP_RELU:
                    સ્થિતિ GGML_UNARY_OP_SIGMOID:
                    સ્થિતિ GGML_UNARY_OP_GELU:
                    સ્થિતિ GGML_UNARY_OP_GELU_QUICK:
                    સ્થિતિ GGML_UNARY_OP_SILU:
                    સ્થિતિ GGML_UNARY_OP_HARDSWISH:
                    સ્થિતિ GGML_UNARY_OP_HARDSIGMOID:
                    સ્થિતિ GGML_UNARY_OP_EXP:
                    સ્થિતિ GGML_UNARY_OP_GELU_ERF:
                    સ્થિતિ GGML_UNARY_OP_SOFTPLUS:
                    સ્થિતિ GGML_UNARY_OP_EXPM1:
                    સ્થિતિ GGML_UNARY_OP_FLOOR:
                    સ્થિતિ GGML_UNARY_OP_CEIL:
                    સ્થિતિ GGML_UNARY_OP_ROUND:
                    સ્થિતિ GGML_UNARY_OP_TRUNC:
                    સ્થિતિ GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        તોડો;
                    મૂળભૂત:
                        તોડો;
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            તોડો;
        સ્થિતિ GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            તોડો;
        સ્થિતિ GGML_OP_IM2COL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            તોડો;
        સ્થિતિ GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_SSM_SCAN:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            તોડો;
        સ્થિતિ GGML_OP_GATED_DELTA_NET:
            {
                અચલ uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            }
            તોડો;
        સ્થિતિ GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            તોડો;
        સ્થિતિ GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            તોડો;
        સ્થિતિ GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            તોડો;
        સ્થિતિ GGML_OP_ટોચ_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            તોડો;
        સ્થિતિ GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            તોડો;
        સ્થિતિ GGML_OP_SUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            તોડો;
        સ્થિતિ GGML_OP_SUM_ROWS:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
            તોડો;
        સ્થિતિ GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            તોડો;
        મૂળભૂત:
            તોડો;
    }
    જો (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize) ||
        (src2 != nullptr &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize)) {
        supports_op = ખોટું;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    જો (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } નહીંતર {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    પાછા supports_op;
}

સ્થિર સંરચના ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ ખાલી,
    /* .buffer_from_host_ptr = */ ખાલી,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ ખાલી,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

સ્થિર અચલ અક્ષર * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    પાછા ctx->name;
}

સ્થિર size_t ggml_backend_webgpu_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    પાછા ctx->device_ગણતરી;
}

// Only one device is supported for now
સ્થિર ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_TOTAL_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    સ્થિર ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    સ્થિર ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_TOTAL_END(reg_get_device, reg_ctx->webgpu_global_ctx);
    પાછા &device;
}

સ્થિર અચલ સંરચના ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_ગણતરી,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ ખાલી,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    સ્થિર ggml_backend_webgpu_reg_context * ctx = નવું ggml_backend_webgpu_reg_context();

    સ્થિર ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_ગણતરી = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    જો (ctx->webgpu_global_ctx != nullptr && ctx->webgpu_global_ctx->instance != nullptr) {
        પાછા &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.data();
    instance_descriptor.required_Feature_ગણતરી                 = instance_features.size();

#જો !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    અચલ અક્ષર * અચલ          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_ગણતરી = 1;
    instance_descriptor.nextInChain        = &instanceTogglesDesc;
#endif

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(નવું webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    જો (ctx->webgpu_global_ctx->instance != nullptr) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    જો (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#જો !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#endif
    ) {
        ctx->device_ગણતરી = 1;
    }

    પાછા &reg;
}

ggml_backend_t ggml_backend_webgpu_init(શૂન્ય) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    જો (ggml_backend_reg_dev_ગણતરી(reg) == 0) {
        પાછા nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    પાછા ggml_backend_webgpu_backend_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
