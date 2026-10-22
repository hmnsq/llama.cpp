/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#অন্তর্ভাব "ggml-webgpu.h"

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-webgpu-shader-lib.hpp"
#অন্তর্ভাব "ggml.h"

#ifdef __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#endif

#অন্তর্ভাব <webgpu/webgpu_cpp.h>

#অন্তর্ভাব <atomic>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstring>
#ifdef GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#endif
#যদি defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#endif
#অন্তর্ভাব <memory>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <optional>
#অন্তর্ভাব <string>
#অন্তর্ভাব <utility>
#অন্তর্ভাব <vector>

#define ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#define CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
স্থির অন্তঃরেখ শূন্য compute_2d_workgroups(uint32_t যোগ_wg, uint32_t max_per_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(যোগ_wg, max_per_dim));
    wg_x = CEIL_DIV(যোগ_wg, wg_y);
}

স্থির অন্তঃরেখ uint32_t ggml_webgpu_u32_from_f32(দশমিক value) {
    uint32_t bits;
    memcpy(&bits, &value, আকার(bits));
    ফেরত bits;
}

#ifdef GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(বার্তা)  std::cout << বার্তা << std::endl
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#অন্যথা
#    define WEBGPU_LOG_DEBUG(বার্তা) ((শূন্য) 0)
#endif  // GGML_WEBGPU_DEBUG

#ifdef GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_যোগ_START(id) auto cpu_যোগ_start##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_যোগ_END(id, ctx)                                                         \
        auto   cpu_যোগ_end##id = std::chrono::high_resolution_clock::now();                            \
        দ্বিগুণ cpu_যোগ_time##id =                                                                      \
            std::chrono::duration<দ্বিগুণ, std::milli>(cpu_যোগ_end##id - cpu_যোগ_start##id).count(); \
        (ctx)->cpu_time_ms[#id] += cpu_যোগ_time##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id) auto cpu_detail_start_##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)                                                          \
        auto   cpu_detail_end_##id = std::chrono::high_resolution_clock::now();                             \
        দ্বিগুণ cpu_detail_time_##id =                                                                       \
            std::chrono::duration<দ্বিগুণ, std::milli>(cpu_detail_end_##id - cpu_detail_start_##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_time_##id;
#অন্যথা
#    define WEBGPU_CPU_PROFILE_যোগ_START(id)
#    define WEBGPU_CPU_PROFILE_যোগ_END(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_START(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_END(id, ctx)
#endif  // GGML_WEBGPU_CPU_PROFILE

#ifdef GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_গুনতি        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_আকার_BYTES (WEBGPU_MAX_PROFILE_QUERY_গুনতি * আকার(uint64_t))
#endif

/* Constants */

#define WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_আকার 64u
#define WEBGPU_অঙ্ক_PARAM_SLOT_SAFETY_MARGIN      10u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#define WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#define WEBGPU_PARAMS_BUF_আকার_BYTES             128  // enough for 32 parameters
#define WEBGPU_SET_ROWS_ERROR_BUF_আকার_BYTES     4
#define WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
স্থির শূন্য * ধ্রুবক webgpu_ptr_base = (শূন্য *) (uintptr_t) 0x1000;  // NOLINT

স্থির size_t ggml_webgpu_tensor_offset(ধ্রুবক ggml_tensor * tensor) {
    ধ্রুবক ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    ফেরত (size_t) ((uintptr_t) base_tensor->তথ্য - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
স্থির শূন্য ggml_webgpu_create_বাফার(wgpu::Device &    device,
                                      wgpu::বাফার_2 &    বাফার,
                                      size_t            size,
                                      wgpu::বাফার_Usage usage,
                                      ধ্রুবক অক্ষর *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
কাঠামো webgpu_param_arena {
    wgpu::বাফার_2 বাফার;
    size_t       slot_stride = 0;
    size_t       slot_আকার   = 0;
    uint32_t     slot_গুনতি  = 0;
    uint32_t     পরবর্তী_slot   = 0;

    শূন্য init(wgpu::Device device, size_t slot_আকার, uint32_t slot_গুনতি, size_t alignment) {
        এই->slot_stride = ROUNDUP_POW2(slot_আকার, alignment);
        এই->slot_আকার   = slot_আকার;
        এই->slot_গুনতি  = slot_গুনতি;
        এই->পরবর্তী_slot   = 0;

        ggml_webgpu_create_বাফার(device, বাফার, এই->slot_stride * slot_গুনতি,
                                  wgpu::বাফার_Usage::CopyDst | wgpu::বাফার_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_slot(size_t size) {
        GGML_ASSERT(size <= slot_আকার);
        যদি (পরবর্তী_slot >= slot_গুনতি) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        ফেরত slot_stride * পরবর্তী_slot++;
    }

    শূন্য reset() { পরবর্তী_slot = 0; }

    শূন্য cleanup() {
        যদি (বাফার) {
            বাফার.Destroy();
            বাফার = nullptr;
        }
    }

    ~webgpu_param_arena() { এই->cleanup(); }
};

কাঠামো webgpu_encoded_op {
    uint32_t অঙ্ক_kernels = 0;
#ifdef GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_names;
#endif
};

কাঠামো webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> bind_group_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

কাঠামো webgpu_capabilities {
    wgpu::Limits limits;
    যৌক্তিক         supports_subgroups       = মিথ্যা;
    যৌক্তিক         supports_subgroup_matrix = মিথ্যা;
    যৌক্তিক         supports_dot_product     = মিথ্যা;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_আকার     = 0;
    uint32_t min_subgroup_আকার = 0;
    uint32_t max_subgroup_আকার = 0;
    size_t   memset_bytes_per_thread;
};

// Stores global webgpu members
কাঠামো webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::Queue    queue;
    uint32_t       command_submit_batch_আকার = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_আকার;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::বাফার_2         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::বাফার_2    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#ifdef GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, দ্বিগুণ> cpu_time_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, দ্বিগুণ> cpu_detail_ms;
#endif

#ifdef GGML_WEBGPU_DEBUG
    wgpu::বাফার_2 debug_host_buf;
    wgpu::বাফার_2 debug_dev_buf;
#endif

    ~webgpu_global_context_struct() {
        যদি (এই->get_tensor_staging_buf) {
            এই->get_tensor_staging_buf.Destroy();
            এই->get_tensor_staging_buf = nullptr;
        }
        যদি (এই->memset_params_buf) {
            এই->memset_params_buf.Destroy();
            এই->memset_params_buf = nullptr;
        }
#ifdef GGML_WEBGPU_DEBUG
        যদি (এই->debug_host_buf) {
            এই->debug_host_buf.Destroy();
            এই->debug_host_buf = nullptr;
        }
        যদি (এই->debug_dev_buf) {
            এই->debug_dev_buf.Destroy();
            এই->debug_dev_buf = nullptr;
        }
#endif
    }
};

প্রকারলেখ std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
কাঠামো webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::বাফার_2             set_rows_dev_error_buf;
    wgpu::বাফার_2             set_rows_host_error_buf;
    wgpu::CommandEncoder     active_command_encoder;
    wgpu::ComputePassEncoder active_compute_pass;
    যৌক্তিক                     batch_compute_passes = সত্য;

    size_t memset_bytes_per_thread;

#ifdef GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, দ্বিগুণ> shader_gpu_time_ms;
    wgpu::বাফার_2                            profile_timestamp_dev_buf;
    wgpu::বাফার_2                            profile_timestamp_host_buf;
    wgpu::QuerySet                          profile_timestamp_query_set;
    uint32_t                                profile_timestamp_query_গুনতি = 0;
#endif

    ~webgpu_context_struct() {
#ifdef GGML_WEBGPU_GPU_PROFILE
        যদি (এই->profile_timestamp_host_buf) {
            এই->profile_timestamp_host_buf.Destroy();
            এই->profile_timestamp_host_buf = nullptr;
        }
        যদি (এই->profile_timestamp_dev_buf) {
            এই->profile_timestamp_dev_buf.Destroy();
            এই->profile_timestamp_dev_buf = nullptr;
        }
        যদি (এই->profile_timestamp_query_set) {
            এই->profile_timestamp_query_set.Destroy();
            এই->profile_timestamp_query_set = nullptr;
        }
#endif
        যদি (এই->set_rows_host_error_buf) {
            এই->set_rows_host_error_buf.Destroy();
            এই->set_rows_host_error_buf = nullptr;
        }
        যদি (এই->set_rows_dev_error_buf) {
            এই->set_rows_dev_error_buf.Destroy();
            এই->set_rows_dev_error_buf = nullptr;
        }
    }
};

প্রকারলেখ std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
কাঠামো ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_গুনতি;
    ধ্রুবক অক্ষর *          name;
};

// Per-device struct for the global logical device interface
কাঠামো ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
কাঠামো ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
কাঠামো ggml_backend_webgpu_বাফার_context {
    wgpu::বাফার_2          বাফার;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_বাফার_context(wgpu::বাফার_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        বাফার(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

স্থির webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ধ্রুবক অক্ষর *                             shader_code,
                                                   ধ্রুবক অক্ষর *                             label,
                                                   ধ্রুবক std::vector<wgpu::ConstantEntry> & constants = {}) {
    wgpu::ShaderSourceWGSL shader_source;
    shader_source.code = shader_code;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.পরবর্তী_In_Chain = &shader_source;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.entryPoint = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = nullptr;  // nullptr means auto layout
    যদি (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.তথ্য();
        pipeline_desc.compute.constant_গুনতি = constants.size();
    }
    ফেরত { device.CreateComputePipeline(&pipeline_desc), label };
}

স্থির শূন্য ggml_webgpu_create_বাফার(wgpu::Device &    device,
                                      wgpu::বাফার_2 &    বাফার,
                                      size_t            size,
                                      wgpu::বাফার_Usage usage,
                                      ধ্রুবক অক্ষর *      label) {
    wgpu::বাফার_Descriptor বাফার_desc;
    বাফার_desc.size             = size;
    বাফার_desc.usage            = usage;
    বাফার_desc.label            = label;
    বাফার_desc.mappedAtCreation = মিথ্যা;

    // TODO: error handling
    বাফার = device.Create_বাফার(&বাফার_desc);
}

স্থির wgpu::বাফার_2 ggml_webgpu_tensor_buf(ধ্রুবক ggml_tensor * tensor) {
    ggml_backend_webgpu_বাফার_context * ctx = (ggml_backend_webgpu_বাফার_context *) tensor->বাফার->context;
    ফেরত ctx->বাফার;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
স্থির size_t ggml_webgpu_tensor_align_offset(ধ্রুবক ggml_tensor * t, size_t alignment) {
    ধ্রুবক size_t offset    = ggml_webgpu_tensor_offset(t);
    ধ্রুবক size_t type_আকার = ggml_type_আকার(t->type);
    size_t       aligned   = offset & ~(alignment - 1);
    যখন ((offset - aligned) % type_আকার != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    ফেরত aligned;
}

স্থির size_t ggml_webgpu_tensor_misalignment(ধ্রুবক ggml_tensor * t, size_t alignment) {
    ফেরত ggml_webgpu_tensor_offset(t) - ggml_webgpu_tensor_align_offset(t, alignment);
}

স্থির size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ধ্রুবক ggml_tensor * t) {
    ফেরত ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment);
}

স্থির size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, ধ্রুবক ggml_tensor * t) {
    ফেরত ggml_webgpu_tensor_align_offset(t, ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment);
}

স্থির size_t ggml_webgpu_tensor_binding_আকার(ধ্রুবক ggml_tensor * t, size_t alignment) {
    ফেরত ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

স্থির size_t ggml_webgpu_tensor_binding_আকার(webgpu_context & ctx, ধ্রুবক ggml_tensor * t) {
    ফেরত ggml_webgpu_tensor_binding_আকার(t, ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment);
}

স্থির যৌক্তিক ggml_webgpu_tensor_binding_overlap(ধ্রুবক webgpu_global_context & global_ctx,
                                               ধ্রুবক ggml_tensor *           a,
                                               ধ্রুবক ggml_tensor *           b) {
    যদি (a->বাফার != b->বাফার) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক size_t alignment = global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment;
    ধ্রুবক size_t a_offset  = ggml_webgpu_tensor_align_offset(a, alignment);
    ধ্রুবক size_t b_offset  = ggml_webgpu_tensor_align_offset(b, alignment);
    ফেরত a_offset < b_offset + ggml_webgpu_tensor_binding_আকার(b, alignment) &&
           b_offset < a_offset + ggml_webgpu_tensor_binding_আকার(a, alignment);
}

স্থির যৌক্তিক ggml_webgpu_tensor_binding_overlap_range(ধ্রুবক webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_বাফার_t         বাফার,
                                                     size_t                        offset,
                                                     size_t                        size) {
    যদি (tensor->বাফার != বাফার) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক size_t alignment     = global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment;
    ধ্রুবক size_t tensor_offset = ggml_webgpu_tensor_align_offset(tensor, alignment);
    ফেরত tensor_offset < offset + size && offset < tensor_offset + ggml_webgpu_tensor_binding_আকার(tensor, alignment);
}

কাঠামো ggml_webgpu_merged_binding_range {
    size_t offset;
    size_t size;
};

স্থির ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_offset = SIZE_MAX;
    size_t merged_end    = 0;

    জন্য (ggml_tensor * tensor : tensors) {
        ধ্রুবক size_t bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
        ধ্রুবক size_t bind_end    = bind_offset + ggml_webgpu_tensor_binding_আকার(ctx, tensor);

        merged_offset = std::min(merged_offset, bind_offset);
        merged_end    = std::max(merged_end, bind_end);
    }

    ফেরত { merged_offset, merged_end - merged_offset };
}

স্থির uint32_t ggml_webgpu_tensor_merged_element_offset(ধ্রুবক ggml_tensor *                      tensor,
                                                         ধ্রুবক ggml_webgpu_merged_binding_range & merged_range) {
    ফেরত (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_আকার(tensor->type));
}

স্থির wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
                                                              wgpu::বাফার_2 বাফার,
                                                              uint64_t     offset,
                                                              uint64_t     size) {
    wgpu::BindGroupEntry entry = {};
    entry.binding              = binding;
    entry.বাফার               = std::move(বাফার);
    entry.offset               = offset;
    entry.size                 = size;
    ফেরত entry;
}

স্থির wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    ফেরত ggml_webgpu_make_bind_group_entry(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_offset(ctx, tensor),
                                             ggml_webgpu_tensor_binding_আকার(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

template <typename T>
স্থির শূন্য ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T                callback_status,
                                                  T                success_status,
                                                  ধ্রুবক অক্ষর *     wait_name,
                                                  ধ্রুবক অক্ষর *     failure_name,
                                                  ধ্রুবক অক্ষর *     callback_message) {
    যদি (wait_status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    যদি (wait_status == wgpu::WaitStatus::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    যদি (callback_status != success_status) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<পূর্ণসংখ্যা>(callback_status),
                   callback_message);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
স্থির uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    ফেরত UINT32_MAX;
}

স্থির uint32_t ggml_backend_webgpu_get_command_submit_batch_আকার() {
    ফেরত WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_আকার;
}

স্থির শূন্য ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    wgpu::QueueWorkDoneStatus callback_status = wgpu::QueueWorkDoneStatus::Error;
    std::string               callback_message;

    ধ্রুবক wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
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

স্থির শূন্য ggml_backend_webgpu_map_বাফার(webgpu_global_context & ctx,
                                           wgpu::বাফার_2 &          বাফার,
                                           wgpu::MapMode           mode,
                                           size_t                  offset,
                                           size_t                  size) {
    wgpu::MapAsyncStatus callback_status = wgpu::MapAsyncStatus::Error;
    std::string          callback_message;

    ধ্রুবক wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        বাফার.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
                        [&callback_status, &callback_message](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                            callback_status  = status;
                            callback_message = std::string(message);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_status(wait_status, callback_status, wgpu::MapAsyncStatus::Success,
                                          "Buffer map wait", "Buffer map", callback_message.c_str());
}

স্থির শূন্য ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ধ্রুবক wgpu::Command_বাফার commands,
                                                uint32_t &                অঙ্ক_inflight_batches) {
    যদি (অঙ্ক_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_queue(ctx->global_ctx);
        অঙ্ক_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    অঙ্ক_inflight_batches++;
}

#ifdef GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
স্থির শূন্য ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_বাফার_To_বাফার(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_আকার());
    wgpu::Command_বাফার commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_বাফার(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.Get_আকার());
    ধ্রুবক দশমিক * debug_তথ্য = (ধ্রুবক দশমিক *) ctx->debug_host_buf.GetConstMappedRange();
    std::cout << "debug[0]: " << debug_তথ্য[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#endif

স্থির webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ধ্রুবক std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            ফলাফল = {};
    std::vector<wgpu::BindGroup> bind_groups;
    std::vector<size_t>          param_offsets;
    ফলাফল.অঙ্ক_kernels = dispatches.size();

    জন্য (size_t i = 0; i < dispatches.size(); i++) {
        ধ্রুবক webgpu_dispatch_desc & dispatch     = dispatches[i];
        ধ্রুবক size_t                 param_আকার   = dispatch.params.size() * আকার(uint32_t);
        ধ্রুবক size_t                 param_offset = ctx->param_arena.alloc_slot(param_আকার);

        std::vector<wgpu::BindGroupEntry> entries            = dispatch.bind_group_entries;
        uint32_t                          params_binding_অঙ্ক = entries.size();
        entries.push_back(ggml_webgpu_make_bind_group_entry(params_binding_অঙ্ক, ctx->param_arena.বাফার, param_offset,
                                                            ctx->param_arena.slot_আকার));

        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout     = dispatch.pipeline.pipeline.GetBindGroupLayout(0);
        bind_group_desc.entry_গুনতি = entries.size();
        bind_group_desc.entries    = entries.তথ্য();
        bind_group_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.CreateBindGroup(&bind_group_desc));
        param_offsets.push_back(param_offset);
    }

    জন্য (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_বাফার(ctx->param_arena.বাফার, param_offsets[i], dispatches[i].params.তথ্য(),
                                           dispatches[i].params.size() * আকার(uint32_t));
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    জন্য (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_গুনতি + 2 <= WEBGPU_MAX_PROFILE_QUERY_গুনতি);
        ধ্রুবক uint32_t query_begin = ctx->profile_timestamp_query_গুনতি++;
        ধ্রুবক uint32_t query_end   = ctx->profile_timestamp_query_গুনতি++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.querySet                    = ctx->profile_timestamp_query_set;
        ts_writes.beginning_Of_Pass_Write_সূচক   = query_begin;
        ts_writes.end_Of_Pass_Write_সূচক         = query_end;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass(&pass_desc);

        pass.SetPipeline(dispatches[i].pipeline.pipeline);
        pass.SetBindGroup(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
        pass.End();
        ফলাফল.pipeline_names.push_back(dispatches[i].pipeline.name);
    }
#অন্যথা
    জন্য (size_t i = 0; i < dispatches.size(); i++) {
        যদি (ctx->batch_compute_passes) {
            ctx->active_compute_pass.SetPipeline(dispatches[i].pipeline.pipeline);
            ctx->active_compute_pass.SetBindGroup(0, bind_groups[i]);
            ctx->active_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second,
                                                        1);
        } অন্যথা {
            wgpu::ComputePassEncoder pass = ctx->active_command_encoder.BeginComputePass();
            pass.SetPipeline(dispatches[i].pipeline.pipeline);
            pass.SetBindGroup(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.first, dispatches[i].workgroups.second, 1);
            pass.End();
        }
    }
#endif

    ফেরত ফলাফল;
}

স্থির webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    ফেরত ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_group_entries), { wg_x, wg_y } },
    });
}

স্থির শূন্য ggml_backend_webgpu_বাফার_memset(webgpu_global_context & ctx,
                                              wgpu::বাফার_2 &          buf,
                                              uint32_t                value,
                                              size_t                  offset,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) offset, (uint32_t) size, value };
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_bind_group_entry(0, buf, 0, buf.Get_আকার()) };
    size_t                            bytes_per_wg =
        ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup * ctx->capabilities.memset_bytes_per_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, bytes_per_wg);

    ctx->queue.Write_বাফার(ctx->memset_params_buf, 0, params.তথ্য(), params.size() * আকার(uint32_t));

    wgpu::BindGroupEntry params_entry = {};
    params_entry.binding              = 1;
    params_entry.বাফার               = ctx->memset_params_buf;
    params_entry.offset               = 0;
    params_entry.size                 = WEBGPU_PARAMS_BUF_আকার_BYTES;
    entries.push_back(params_entry);

    wgpu::BindGroupDescriptor bind_group_desc;
    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
    bind_group_desc.entry_গুনতি = entries.size();
    bind_group_desc.entries    = entries.তথ্য();
    bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.SetPipeline(ctx->memset_pipeline.pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.End();

    wgpu::Command_বাফার              command  = encoder.Finish();
    std::vector<wgpu::Command_বাফার> commands = { command };
    ctx->queue.Submit(commands.size(), commands.তথ্য());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

স্থির ধ্রুবক অক্ষর * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    ফেরত ctx->name.c_str();
}

স্থির শূন্য ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#ifdef GGML_WEBGPU_CPU_PROFILE
    std::cout << "\n[ggml_webgpu cpu profiling summary]\n";
    দ্বিগুণ যোগ_cpu = 0.0;
    জন্য (ধ্রুবক auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        যোগ_cpu += kv.second;
    }
    std::cout << "ggml_webgpu: total cpu time: " << যোগ_cpu << " ms\n";
    std::cout << "ggml_webgpu: cpu breakdown:\n";
    জন্য (ধ্রুবক auto & kv : ctx->webgpu_ctx->global_ctx->cpu_time_ms) {
        দ্বিগুণ pct = (যোগ_cpu > 0.0) ? (kv.second / যোগ_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    যদি (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::cout << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    জন্য (ধ্রুবক auto & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        দ্বিগুণ pct = (যোগ_cpu > 0.0) ? (kv.second / যোগ_cpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#endif

#ifdef GGML_WEBGPU_GPU_PROFILE
    std::cout << "\n[ggml_webgpu gpu profiling summary]\n";
    দ্বিগুণ যোগ_gpu = 0.0;
    জন্য (ধ্রুবক auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        যোগ_gpu += kv.second;
    }
    std::cout << "ggml_webgpu: total gpu time (all shaders): " << যোগ_gpu << " ms\n";
    std::cout << "\nggml_webgpu: gpu breakdown:\n";
    জন্য (ধ্রুবক auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
        দ্বিগুণ pct = (যোগ_gpu > 0.0) ? (kv.second / যোগ_gpu * 100.0) : 0.0;
        std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#endif

#যদি defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::cout << "ggml_webgpu: gpu/cpu ratio: " << (যোগ_cpu > 0.0 ? যোগ_gpu / যোগ_cpu : 0.0) << "\n";
#endif

    delete ctx;
    delete backend;
}

স্থির webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_আকার(src->type)), (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)), (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_আকার(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
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
    uint32_t যোগ_wg = CEIL_DIV(ne, decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ধ্রুবক যৌক্তিক inplace   = decisions->inplace;

    ধ্রুবক uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    ধ্রুবক uint32_t dst_type_আকার = (uint32_t) ggml_type_আকার(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (((ধ্রুবক int32_t *) dst->op_params)[3] / dst_type_আকার),

        (uint32_t) (src1->nb[0] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),

        1u,
        (uint32_t) (((ধ্রুবক int32_t *) dst->op_params)[0] / dst_type_আকার),
        (uint32_t) (((ধ্রুবক int32_t *) dst->op_params)[1] / dst_type_আকার),
        (uint32_t) (((ধ্রুবক int32_t *) dst->op_params)[2] / dst_type_আকার),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;
    uint32_t                          binding_সূচক = 0;
    যদি (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        binding_সূচক++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_সূচক, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_সূচক + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ne, decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ধ্রুবক uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)),
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
    uint32_t যোগ_wg = CEIL_DIV(ne, decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_আকার;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
    };

    ধ্রুবক uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_আকার);
    ধ্রুবক uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    ধ্রুবক int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ধ্রুবক int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ধ্রুবক int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ধ্রুবক int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ধ্রুবক int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ধ্রুবক int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),

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
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
স্থির webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    ধ্রুবক int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ধ্রুবক int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ধ্রুবক int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ধ্রুবক int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ধ্রুবক int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ধ্রুবক int32_t d1 = ggml_get_op_params_i32(dst, 5);

    // Scalar params matching conv2d_dw.wgsl (weight src0 [KW,KH,1,C], input src1, output dst).
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),

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
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ধ্রুবক যৌক্তিক      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    ধ্রুবক int32_t s0    = ggml_get_op_params_i32(dst, 0);
    ধ্রুবক int32_t s1    = ggml_get_op_params_i32(dst, 1);
    ধ্রুবক int32_t p0    = ggml_get_op_params_i32(dst, 2);
    ধ্রুবক int32_t p1    = ggml_get_op_params_i32(dst, 3);
    ধ্রুবক int32_t d0    = ggml_get_op_params_i32(dst, 4);
    ধ্রুবক int32_t d1    = ggml_get_op_params_i32(dst, 5);
    ধ্রুবক যৌক্তিক    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    ধ্রুবক uint32_t KW = src0->ne[0];
    ধ্রুবক uint32_t KH = is_2D ? src0->ne[1] : 1;
    ধ্রুবক uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    ধ্রুবক uint32_t IW = src1->ne[0];
    ধ্রুবক uint32_t IH = is_2D ? src1->ne[1] : 1;
    ধ্রুবক uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    ধ্রুবক uint32_t OW = dst->ne[1];
    ধ্রুবক uint32_t OH = is_2D ? dst->ne[2] : 1;

    ধ্রুবক uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_আকার(src1->type));
    ধ্রুবক uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)) : 0;
    ধ্রুবক uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type));
    ধ্রুবক uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type));

    ধ্রুবক uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_আকার(dst->type));
    ধ্রুবক uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type));
    ধ্রুবক uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)) : 0;
    ধ্রুবক uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),

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
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ধ্রুবক uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->tokens_per_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),

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

    ধ্রুবক uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_আকার_2);
    ধ্রুবক uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_আকার        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    যদি (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    যদি (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    যদি (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_scan_pipeline(shader_lib_ctx, xbc_overlap);
    auto *          decisions = static_cast<ggml_webgpu_ssm_scan_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t offset_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type));
    uint32_t offset_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_আকার(src2->type));
    uint32_t offset_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_আকার(src3->type));
    uint32_t offset_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_আকার(src4->type));
    uint32_t offset_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_আকার(src5->type));
    uint32_t offset_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_আকার(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    যদি (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
    } অন্যথা যদি (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    } অন্যথা যদি (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        offset_x         = ggml_webgpu_tensor_merged_element_offset(src1, xbc_merged_range);
        offset_B         = ggml_webgpu_tensor_merged_element_offset(src4, xbc_merged_range);
        offset_C         = ggml_webgpu_tensor_merged_element_offset(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_offset = 0;
    size_t xbc_bind_আকার   = 0;
    যদি (xbc_overlap > 0) {
        xbc_bind_offset = xbc_merged_range.offset;
        xbc_bind_আকার   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        offset_x,
        offset_dt,
        offset_A,
        offset_B,
        offset_C,
        offset_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_আকার(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_আকার(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_আকার(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_আকার(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_আকার(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_আকার(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_আকার(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_আকার(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_আকার(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_অঙ্ক = 0;

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, src0),
    };
    // xbc_merged binding
    যদি (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_অঙ্ক++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_offset, xbc_bind_আকার));
    }
    // x
    যদি (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, src1));
    }
    // B
    যদি (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, src4));
    }
    // C
    যদি (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক++, dst));

    ধ্রুবক uint32_t যোগ_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ধ্রুবক uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(যোগ_wg, max_wg_per_dim, wg_x, wg_y);

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ধ্রুবক uint32_t s_v      = (uint32_t) src2->ne[0];
    ধ্রুবক uint32_t h        = (uint32_t) src2->ne[1];
    ধ্রুবক uint32_t n_tokens = (uint32_t) src2->ne[2];
    ধ্রুবক uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ধ্রুবক uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ধ্রুবক দশমিক    scale    = 1.0f / sqrtf((দশমিক) s_v);
    uint32_t       scale_u32;
    memcpy(&scale_u32, &scale, আকার(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_tokens,
        n_seqs,
        s_v * h * n_tokens * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_আকার(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_আকার(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_আকার(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_আকার(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_আকার(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_আকার(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_আকার(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_আকার(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0), ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2), ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4), ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, dst),
    };

    যদি (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst_fuse));
    }

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

স্থির std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    যদি (ggml_is_empty(src) || ggml_is_empty(idx)) {
        ফেরত std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_set_rows_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_set_rows_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_আকার(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)), (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)), (uint32_t) (idx->nb[0] / ggml_type_আকার(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_আকার(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_আকার(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::BindGroupEntry> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));

    যদি (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(3, ctx->set_rows_dev_error_buf, 0,
                                                            ctx->set_rows_dev_error_buf.Get_আকার()));
    }

    uint32_t threads;
    যদি (ggml_is_quantized(dst->type)) {
        ধ্রুবক uint32_t blocks_per_row = src->ne[0] / ggml_blck_আকার(dst->type);
        threads =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_blocks ? (blocks_per_row / 2) : blocks_per_row);
    } অন্যথা যদি (decisions->vec4) {
        threads = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } অন্যথা {
        threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(threads, decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
স্থির std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_আকার_entry(uint32_t wg_আকার) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].চাবি   = "wg_size";
    constants[0].value = wg_আকার;
    ফেরত constants;
}

স্থির webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ধ্রুবক যৌক্তিক float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ধ্রুবক uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type));
    ধ্রুবক uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type));
    ধ্রুবক uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_আকার(src->type));
    ধ্রুবক uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_আকার(src->type));
    ধ্রুবক uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_আকার(src->type));

    ধ্রুবক যৌক্তিক vec4_aligned = offset_src % 4 == 0 && offset_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_rows_pipeline(shader_lib_ctx, vec4_aligned);
    auto *          decisions = static_cast<ggml_webgpu_get_rows_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { offset_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_আকার(idx->type)),
                                     offset_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_আকার(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_আকার(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_আকার(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
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
    uint32_t যোগ_rows     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t যোগ_threads  = float_parallel ? blocks_per_row * যোগ_rows : যোগ_rows;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(যোগ_threads, decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির শূন্য ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_আকার        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ধ্রুবক size_t dst_offset           = ggml_webgpu_tensor_offset(dst);
    ধ্রুবক size_t q8_src1_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment);
    ধ্রুবক size_t q8_src1_binding_আকার = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> q8_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_offset, q8_src1_binding_আকার)
    };

    auto q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_আকার     = q8_decisions->wg_আকার;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ধ্রুবক uint32_t wg_per_vec     = (src0->ne[0] / 4 + (q8_wg_আকার - 1)) / q8_wg_আকার;
    ধ্রুবক uint32_t q8_যোগ_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_per_vec;
    ধ্রুবক uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(q8_যোগ_wg, max_wg_per_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

স্থির webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    যৌক্তিক use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    যৌক্তিক use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_product,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_আকার              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_আকার        = ctx->global_ctx->capabilities.min_subgroup_আকার;
    shader_lib_ctx.max_subgroup_আকার        = ctx->global_ctx->capabilities.max_subgroup_আকার;
    shader_lib_ctx.supports_dot_product     = ctx->global_ctx->capabilities.supports_dot_product;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    ধ্রুবক যৌক্তিক src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    যদি (use_mat_vec) {
        যদি (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } অন্যথা {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_আকার   = 0;
    যদি (src_overlap) {
        ধ্রুবক ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_আকার   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { offset_src0,
                                     offset_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::BindGroupEntry> entries = {};
    যদি (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        auto & mmvq_qq8_entry = dispatches[0].bind_group_entries[1];
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_entry.offset,
                                                            mmvq_qq8_entry.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    } অন্যথা যদি (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_আকার));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } অন্যথা {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ধ্রুবক uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;

    যদি (use_mat_vec) {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
        uint32_t যোগ_wg      = output_groups * batches;
        compute_2d_workgroups(যোগ_wg, max_wg_per_dim, wg_x, wg_y);
    } অন্যথা {
        auto * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        যদি (decisions->use_subgroup_matrix) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_matrix_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_matrix_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } অন্যথা {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_আকার_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_আকার_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t যোগ_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(যোগ_wg, max_wg_per_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    ফেরত ggml_backend_webgpu_build_multi(ctx, dispatches);
}

স্থির webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    ধ্রুবক uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ধ্রুবক uint32_t param_n_expert_used = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_আকার(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
    };

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_আকার(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_আকার(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_আকার(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_আকার(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ধ্রুবক uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_per_wg);
    uint32_t       যোগ_wg       = output_groups * param_n_expert_used;
    compute_2d_workgroups(যোগ_wg, max_wg_per_dim, wg_x, wg_y);

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
স্থির uint32_t ggml_webgpu_mul_mat_id_max_tiles(ধ্রুবক ggml_tensor * dst) {
    ধ্রুবক uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    ধ্রুবক uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    ফেরত CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_আকার_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
স্থির size_t ggml_webgpu_mul_mat_id_tiles_nbytes(ধ্রুবক ggml_tensor * dst) {
    ফেরত (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * আকার(uint32_t);
}

স্থির webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    যদি (dst->ne[2] == 1) {
        ফেরত ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ধ্রুবক uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ধ্রুবক uint32_t param_n_expert_used = (uint32_t) dst->ne[1];
    ধ্রুবক uint32_t param_n_tokens      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_আকার(src2->type)),
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) (src2->nb[1] / ggml_type_আকার(src2->type)),
    };

    ধ্রুবক size_t dst_offset       = ggml_webgpu_tensor_offset(dst);
    ধ্রুবক size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * আকার(uint32_t);
    ধ্রুবক size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ধ্রুবক size_t ids_cache_align_offset = ROUNDUP_POW2(
        dst_offset + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment);
    ধ্রুবক size_t ids_tiles_align_offset =
        ROUNDUP_POW2(ids_cache_align_offset + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment);

    ধ্রুবক size_t ids_cache_bind_আকার = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ধ্রুবক size_t ids_tiles_bind_আকার = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::BindGroupEntry> gather_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_offset(ctx, src2),
                                          ggml_webgpu_tensor_binding_আকার(ctx, src2)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_আকার),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_আকার),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_used,
        param_n_tokens,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::BindGroupEntry> main_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
                                          ggml_webgpu_tensor_binding_আকার(ctx, src0)),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_offset(ctx, src1),
                                          ggml_webgpu_tensor_binding_আকার(ctx, src1)),
        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                          ggml_webgpu_tensor_binding_আকার(ctx, dst)),
        ggml_webgpu_make_bind_group_entry(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_offset, ids_cache_bind_আকার),
        ggml_webgpu_make_bind_group_entry(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_offset, ids_tiles_bind_আকার),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    auto * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_আকার_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_আকার_N);

    ধ্রুবক uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_আকার_m;
    ধ্রুবক uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ধ্রুবক uint32_t যোগ_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    ফেরত ggml_backend_webgpu_build_multi(ctx, dispatches);
}

কাঠামো ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::BindGroupEntry> entries;
    size_t                            kv_bind_offset = 0;
    size_t                            kv_bind_আকার   = 0;
    যৌক্তিক                              has_mask       = মিথ্যা;
    যৌক্তিক                              has_sinks      = মিথ্যা;
    যৌক্তিক                              kv_overlap     = মিথ্যা;
};

স্থির যৌক্তিক ggml_webgpu_flash_attn_use_vec_path(ধ্রুবক webgpu_global_context & global_ctx,
                                                ধ্রুবক ggml_tensor *           Q,
                                                ধ্রুবক ggml_tensor *           K,
                                                ধ্রুবক ggml_tensor *           V) {
    ধ্রুবক size_t storage_offset_alignment = global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment;

    ধ্রুবক যৌক্তিক k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_offset_alignment);
    ধ্রুবক যৌক্তিক v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_offset_alignment);

    ধ্রুবক uint32_t k_vec_শীর্ষ_align =
        ggml_is_quantized(K->type) ? ggml_blck_আকার(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    ধ্রুবক uint32_t v_vec_শীর্ষ_align =
        ggml_is_quantized(V->type) ? ggml_blck_আকার(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
    ধ্রুবক যৌক্তিক kv_vec_শীর্ষ_dims_aligned = Q->ne[0] % k_vec_শীর্ষ_align == 0 && V->ne[0] % v_vec_শীর্ষ_align == 0;

    ফেরত global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_শীর্ষ_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

স্থির ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    mask,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    দশমিক scale         = ggml_get_op_params_f32(dst, 0);
    দশমিক max_bias      = ggml_get_op_params_f32(dst, 1);
    দশমিক logit_softcap = ggml_get_op_params_f32(dst, 2);
    যদি (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    দশমিক n_শীর্ষ_log2 = দশমিক(1u << (uint32_t) floor(log2(Q->ne[2])));
    দশমিক m0          = powf(2.0f, -(max_bias) / n_শীর্ষ_log2);
    দশমিক m1          = powf(2.0f, -(max_bias / 2.0f) / n_শীর্ষ_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = mask;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_matrix = ctx->global_ctx->capabilities.supports_subgroup_matrix;
    op.shader_lib_ctx.max_wg_আকার              = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    op.shader_lib_ctx.wg_mem_limit_bytes       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_আকার;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_আকার        = ctx->global_ctx->capabilities.min_subgroup_আকার;
    op.shader_lib_ctx.max_subgroup_আকার        = ctx->global_ctx->capabilities.max_subgroup_আকার;

    op.has_mask   = mask != nullptr;
    op.has_sinks  = sinks != nullptr;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t offset_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_আকার(K->type));
    uint32_t offset_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_আকার(V->type));
    যদি (op.kv_overlap) {
        ধ্রুবক ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_offset                                   = merged_range.offset;
        op.kv_bind_আকার                                     = merged_range.size;
        offset_k                                            = ggml_webgpu_tensor_merged_element_offset(K, merged_range);
        offset_v                                            = ggml_webgpu_tensor_merged_element_offset(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_আকার(Q->type)),
        offset_k,
        offset_v,
        op.has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_আকার(mask->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_আকার(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_আকার(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_আকার(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_আকার(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_আকার(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_আকার(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_আকার(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_আকার(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_আকার(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_আকার(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_mask ? (uint32_t) (mask->nb[3] / ggml_type_আকার(mask->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_শীর্ষ_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
    };
    যদি (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_আকার));
    } অন্যথা {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
    }
    uint32_t binding_সূচক = op.kv_overlap ? 2u : 3u;
    যদি (op.has_mask) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_সূচক++, mask));
    }
    যদি (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_সূচক++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_সূচক++, dst));

    ফেরত op;
}

স্থির uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ধ্রুবক uint64_t kv_span = (uint64_t) kv_tile;
    যখন ((2u * nwg * kv_span) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    ফেরত std::min(nwg, vec_nwg_cap);
}

স্থির webgpu_encoded_op ggml_webgpu_flash_attn_direct(webgpu_context & ctx, ধ্রুবক ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_per_শীর্ষ = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_per_শীর্ষ * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

স্থির webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             mask,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    auto *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::বাফার_2 blk_buf         = {};
    uint64_t     blk_আকার_bytes  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_গুনতি = 0;

    ধ্রুবক uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_আকার;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ধ্রুবক uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ধ্রুবক যৌক্তিক     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_base  = 0;
    uint64_t     tmp_আকার_bytes  = 0;
    wgpu::বাফার_2 tmp_buf         = {};
    uint64_t     tmp_bind_offset = 0;
    uint64_t     tmp_bind_আকার   = 0;
    ধ্রুবক size_t align_bytes     = ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment;
    ধ্রুবক size_t dst_offset      = ggml_webgpu_tensor_offset(dst);
    size_t       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);

    যদি (use_vec_reduce) {
        ধ্রুবক uint64_t tmp_তথ্য_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ধ্রুবক uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_base                 = tmp_তথ্য_elems;
        tmp_আকার_bytes =
            ROUNDUP_POW2((tmp_তথ্য_elems + tmp_stats_elems) * আকার(দশমিক), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_আকার   = tmp_আকার_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_আকার_bytes, align_bytes);
    } অন্যথা {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_আকার_bytes  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_offset = scratch_offset;
        tmp_bind_আকার   = tmp_আকার_bytes;
        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_আকার_bytes, align_bytes);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::BindGroupEntry> blk_entries;
    যদি (op.has_mask) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ধ্রুবক uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_আকার(mask->type));
        blk_batch_গুনতি             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ধ্রুবক uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_গুনতি;
        blk_আকার_bytes              = ROUNDUP_POW2(blk_elems * আকার(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        ধ্রুবক ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_আকার(mask->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(mask),
                                              ggml_webgpu_tensor_align_offset(ctx, mask),
                                              ggml_webgpu_tensor_binding_আকার(ctx, mask)),
            ggml_webgpu_make_bind_group_entry(1, blk_buf, scratch_offset, blk_আকার_bytes),
        };
        scratch_offset = ROUNDUP_POW2(scratch_offset + blk_আকার_bytes, align_bytes);
    }

    std::vector<uint32_t> split_params = op.params;
    যদি (op.has_mask) {
        split_params.push_back(0u);                     // blk_base
        split_params.push_back(blk_nblk0);              // blk_nblk0
        split_params.push_back(blk_nblk1);              // blk_nblk1
    }
    split_params.push_back(0u);                         // tmp_data_base
    split_params.push_back((uint32_t) tmp_stats_base);  // tmp_stats_base
    split_params.push_back(nwg);                        // nwg

    std::vector<wgpu::BindGroupEntry> split_entries = {
        ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_offset(ctx, Q),
                                          ggml_webgpu_tensor_binding_আকার(ctx, Q)),
    };
    যদি (op.kv_overlap) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), op.kv_bind_offset, op.kv_bind_আকার));
    } অন্যথা {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
                                                                  ggml_webgpu_tensor_binding_আকার(ctx, K)));
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
                                                                  ggml_webgpu_tensor_binding_আকার(ctx, V)));
    }
    uint32_t split_binding_সূচক = op.kv_overlap ? 2u : 3u;
    যদি (op.has_mask) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_সূচক++, ggml_webgpu_tensor_buf(mask),
                                                                  ggml_webgpu_tensor_align_offset(ctx, mask),
                                                                  ggml_webgpu_tensor_binding_আকার(ctx, mask)));
    }
    যদি (op.has_sinks) {
        split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_সূচক++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_offset(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_আকার(ctx, sinks)));
    }
    যদি (op.has_mask) {
        split_entries.push_back(
            ggml_webgpu_make_bind_group_entry(split_binding_সূচক++, blk_buf, blk_entries[1].offset, blk_আকার_bytes));
    }
    split_entries.push_back(
        ggml_webgpu_make_bind_group_entry(split_binding_সূচক++, tmp_buf, tmp_bind_offset, tmp_bind_আকার));
    split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_সূচক++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_offset(ctx, dst),
                                                              ggml_webgpu_tensor_binding_আকার(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::BindGroupEntry> reduce_entries;
    যদি (use_vec_reduce) {
        ধ্রুবক uint32_t reduce_sg_আকার = ctx->global_ctx->capabilities.max_subgroup_আকার;
        ধ্রুবক uint32_t reduce_wg_আকার = std::max(
            reduce_sg_আকার,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_আকার,
                                          ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_আকার                    = reduce_wg_আকার;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_base,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_group_entry(0, tmp_buf, tmp_bind_offset, tmp_আকার_bytes),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_offset(ctx, dst),
                                              ggml_webgpu_tensor_binding_আকার(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ধ্রুবক uint64_t split_wg_যোগ = (uint64_t) wg_x * nwg;
    GGML_ASSERT(split_wg_যোগ <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    যদি (op.has_mask) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_গুনতি }
        });
    }
    dispatches.push_back({
        pipeline, std::move(split_params), std::move(split_entries), { (uint32_t) split_wg_যোগ, 1u }
    });
    যদি (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    ফেরত ggml_backend_webgpu_build_multi(ctx, dispatches);
}

স্থির webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    mask,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, mask, sinks, dst);
    যদি (ggml_webgpu_flash_attn_use_vec_path(ctx->global_ctx, Q, K, V)) {
        ফেরত ggml_webgpu_flash_attn_vec(ctx, Q, K, V, mask, sinks, dst, std::move(op));
    }
    ফেরত ggml_webgpu_flash_attn_direct(ctx, op);
}

স্থির webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    যৌক্তিক is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ধ্রুবক যৌক্তিক inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_আকার(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    যদি (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        নির্বাচন (unary_op) {
            ক্ষেত্র GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    দশমিক alpha_n = ggml_get_op_params_f32(dst, 1);
                    দশমিক alpha_p = ggml_get_op_params_f32(dst, 2);
                    দশমিক beta    = ggml_get_op_params_f32(dst, 3);
                    দশমিক eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    বিরতি;
                }
            পূর্বনির্ধারিত:
                বিরতি;
        }
    } অন্যথা যদি (dst->op == GGML_OP_CLAMP) {
        দশমিক clamp_min = ggml_get_op_params_f32(dst, 0);
        দশমিক clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } অন্যথা যদি (dst->op == GGML_OP_FILL) {
        দশমিক fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::BindGroupEntry> entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, effective_src),
    };
    যদি (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_দ্বিমিক_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ধ্রুবক যৌক্তিক      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_দ্বিমিক_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_দ্বিমিক_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
    size_t src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_আকার   = 0;
    যদি (decisions->src_overlap) {
        ধ্রুবক ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_আকার   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::BindGroupEntry> entries;

    যদি (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_আকার));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } অন্যথা {
        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_আকার(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_offset,
                                                            ggml_webgpu_tensor_binding_আকার(ctx, src1)));
        যদি (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_add_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_add_id_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_আকার(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_আকার(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_আকার(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::BindGroupEntry> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));

    যদি (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       যোগ_wg       = ggml_nrows(dst);
    ধ্রুবক uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(যোগ_wg, max_wg_per_dim, wg_x, wg_y);

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    যদি (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        ফেরত {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ধ্রুবক যৌক্তিক      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_দ্বিমিক_shader_decisions *>(pipeline.context.get());

    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type));
    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type));
    size_t   merged_offset = 0;
    size_t   merged_আকার   = 0;
    যদি (decisions->src_overlap) {
        যদি (ggml_nbytes(src0) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src1);
            merged_আকার   = ggml_webgpu_tensor_binding_আকার(ctx, src1);
            offset_src0   = 0;
            offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type));
        } অন্যথা যদি (ggml_nbytes(src1) == 0) {
            merged_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
            merged_আকার   = ggml_webgpu_tensor_binding_আকার(ctx, src0);
            offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type));
            offset_src1   = 0;
        } অন্যথা {
            ধ্রুবক ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_offset = merged_range.offset;
            merged_আকার   = merged_range.size;
            offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
            offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     offset_src0,
                                     offset_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_আকার(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::BindGroupEntry> entries = {};
    যদি (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_আকার));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } অন্যথা {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ne, decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_আকার(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
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
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ne, decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    যদি (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } অন্যথা যদি (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } অন্যথা {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t offset_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_আকার(rn_src->type));
    uint32_t offset_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_আকার(mul_src->type));
    size_t merged_offset = 0;
    size_t merged_আকার   = 0;

    std::vector<uint32_t> params = {
        offset_rn_src,
        offset_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_আকার(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_আকার(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_আকার(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_আকার(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_আকার(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_আকার(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
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
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ধ্রুবক যৌক্তিক      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    auto *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    যদি (decisions->src_overlap) {
        ধ্রুবক ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_offset  = merged_range.offset;
        merged_আকার    = merged_range.size;
        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
        params[0]      = offset_rn_src;
        params[1]      = offset_mul_src;
    }

    যদি (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
    } অন্যথা যদি (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_আকার));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    } অন্যথা {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
    যদি (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x,
                          wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ধ্রুবক যৌক্তিক inplace         = decisions->inplace;
    ধ্রুবক পূর্ণসংখ্যা  has_freq_factor = (src2 != nullptr);

    ধ্রুবক পূর্ণসংখ্যা n_dims     = ((int32_t *) dst->op_params)[1];
    ধ্রুবক পূর্ণসংখ্যা mode       = ((int32_t *) dst->op_params)[2];
    ধ্রুবক পূর্ণসংখ্যা n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ধ্রুবক পূর্ণসংখ্যা n_offs     = ((int32_t *) dst->op_params)[15];

    দশমিক freq_base;
    দশমিক freq_scale;
    দশমিক ext_factor;
    দশমিক attn_factor;
    দশমিক beta_fast;
    দশমিক beta_slow;
    memcpy(&freq_base, (int32_t *) dst->op_params + 5, আকার(দশমিক));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, আকার(দশমিক));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, আকার(দশমিক));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, আকার(দশমিক));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, আকার(দশমিক));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, আকার(দশমিক));

    পূর্ণসংখ্যা sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * আকার(পূর্ণসংখ্যা));

    দশমিক theta_scale = powf(freq_base, -2.0f / n_dims);

    দশমিক corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)),
        src2 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_আকার(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
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
    যদি (has_freq_factor) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
    }
    যদি (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    ধ্রুবক যৌক্তিক      src_overlap = src1 != nullptr && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    auto * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ধ্রুবক পূর্ণসংখ্যা split = (src1 != nullptr);

    uint32_t offset_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type));
    uint32_t offset_src1 =
        src1 != nullptr ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)) : 0;
    size_t merged_offset = 0;
    size_t merged_আকার   = 0;
    যদি (decisions->src_overlap) {
        ধ্রুবক ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_offset = merged_range.offset;
        merged_আকার   = merged_range.size;
        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        offset_src0,
        offset_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        src1 != nullptr ? (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
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
    যদি (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_আকার));
    } অন্যথা {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
    }
    যদি (split && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };

    যদি (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ধ্রুবক যৌক্তিক inplace     = decisions->inplace;
    ধ্রুবক পূর্ণসংখ্যা  has_mask    = (src1 != nullptr);
    ধ্রুবক পূর্ণসংখ্যা  has_sink    = (src2 != nullptr);
    দশমিক      max_bias    = ggml_get_op_params_f32(dst, 1);
    দশমিক      n_শীর্ষ_log2 = দশমিক(1u << (uint32_t) floor(log2(src0->ne[2])));
    দশমিক      m0          = powf(2.0f, -(max_bias) / n_শীর্ষ_log2);
    দশমিক      m1          = powf(2.0f, -(max_bias / 2.0f) / n_শীর্ষ_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_আকার(src0->type)),
        has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_আকার(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_আকার(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_আকার(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_আকার(src0->type)),
        has_mask ? (uint32_t) (src1->nb[1] / ggml_type_আকার(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[2] / ggml_type_আকার(src1->type)) : 0,
        has_mask ? (uint32_t) (src1->nb[3] / ggml_type_আকার(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_mask ? (uint32_t) src1->ne[2] : 0,
        has_mask ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_শীর্ষ_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::BindGroupEntry> entries     = { ggml_webgpu_make_bind_group_entry(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_offset(ctx, src0),
        ggml_webgpu_tensor_binding_আকার(ctx, src0)) };
    uint32_t                          binding_অঙ্ক = 1;
    যদি (has_mask) {
        entries.push_back(ggml_webgpu_make_bind_group_entry(binding_অঙ্ক, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_offset(ctx, src1),
                                                            ggml_webgpu_tensor_binding_আকার(ctx, src1)));
        binding_অঙ্ক++;
    }
    যদি (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক, src2));
        binding_অঙ্ক++;
    }
    যদি (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_অঙ্ক, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = ggml_nrows(dst);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

স্থির webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    যৌক্তিক is_শীর্ষ_k = dst->op == GGML_OP_শীর্ষ_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_আকার;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    auto * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_merge_pipeline = ctx->shader_lib->get_argsort_merge_pipeline(shader_lib_ctx);

    ধ্রুবক uint32_t src_ne0 = (uint32_t) src->ne[0];
    ধ্রুবক uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ধ্রুবক uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_আকার);
    ধ্রুবক uint32_t block_আকার_2 =
        is_শীর্ষ_k ? std::min(argsort_decisions->wg_আকার, (uint32_t) dst->ne[0]) : argsort_decisions->wg_আকার;
    uint32_t out_ne0 = src_ne0;
    যদি (is_শীর্ষ_k) {
        যদি (npr > 1) {
            ধ্রুবক uint32_t last_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_আকার;
            out_ne0                  = (npr - 1) * block_আকার_2 + std::min(last_tile, block_আকার_2);
        } অন্যথা {
            out_ne0 = block_আকার_2;
        }
    }

    uint32_t merge_len    = block_আকার_2;
    uint32_t merge_passes = 0;
    যখন (merge_len < out_ne0) {
        merge_len <<= 1;
        merge_passes++;
    }

    ধ্রুবক যৌক্তিক start_in_tmp = (merge_passes % 2) == 1;

    ধ্রুবক size_t dst_offset = ggml_webgpu_tensor_offset(dst);
    ধ্রুবক size_t idx_nbytes = out_ne0 * ggml_nrows(dst) * আকার(int32_t);
    ধ্রুবক size_t tmp_offset =
        ROUNDUP_POW2(dst_offset + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment);
    ধ্রুবক size_t tmp_binding_আকার = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ধ্রুবক size_t dst_binding_আকার =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ধ্রুবক uint32_t offset_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type));
    ধ্রুবক uint32_t offset_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type));
    ধ্রুবক uint32_t offset_tmp  = 0;
    ধ্রুবক uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_আকার(src->type));
    ধ্রুবক uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_আকার(src->type));
    ধ্রুবক uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_আকার(src->type));
    ধ্রুবক uint32_t stride_idx1 = out_ne0;
    ধ্রুবক uint32_t stride_idx2 = out_ne0 * (uint32_t) dst->ne[1];
    ধ্রুবক uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ধ্রুবক uint32_t init_offset       = start_in_tmp ? offset_tmp : offset_dst;
    ধ্রুবক size_t   init_align_offset = start_in_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
    ধ্রুবক size_t   init_binding_আকার = start_in_tmp ? tmp_binding_আকার : dst_binding_আকার;

    std::vector<uint32_t> init_params = {
        offset_src,  init_offset, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], out_ne0,
        block_আকার_2,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ধ্রুবক uint32_t যোগ_wg_init  = npr * nrows;
    ধ্রুবক uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    compute_2d_workgroups(যোগ_wg_init, max_wg_per_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::BindGroupEntry> init_entries = {
        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), init_align_offset, init_binding_আকার)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    যদি (merge_passes == 0) {
        ফেরত ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    যৌক্তিক     in_is_tmp = start_in_tmp;
    uint32_t len       = block_আকার_2;
    যখন (len < out_ne0) {
        ধ্রুবক uint32_t nm = CEIL_DIV(out_ne0, 2 * len);

        ধ্রুবক যৌক্তিক     out_is_tmp  = !in_is_tmp;
        ধ্রুবক uint32_t offset_in   = in_is_tmp ? offset_tmp : offset_dst;
        ধ্রুবক uint32_t offset_out  = out_is_tmp ? offset_tmp : offset_dst;
        ধ্রুবক size_t   align_in    = in_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        ধ্রুবক size_t   align_out   = out_is_tmp ? tmp_offset : ggml_webgpu_tensor_align_offset(ctx, dst);
        ধ্রুবক size_t   আকার_in     = in_is_tmp ? tmp_binding_আকার : dst_binding_আকার;
        ধ্রুবক size_t   আকার_out    = out_is_tmp ? tmp_binding_আকার : dst_binding_আকার;
        ধ্রুবক uint32_t শীর্ষ_k_out   = (is_শীর্ষ_k && nm == 1) ? (uint32_t) dst->ne[0] : out_ne0;
        ধ্রুবক uint32_t stride_out1 = শীর্ষ_k_out;
        ধ্রুবক uint32_t stride_out2 = শীর্ষ_k_out * (uint32_t) dst->ne[1];
        ধ্রুবক uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

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
                                               শীর্ষ_k_out,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::BindGroupEntry> merge_entries = {
            ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), align_in, আকার_in),
            ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(dst), align_out, আকার_out)
        };

        uint32_t       wg_x_merge;
        uint32_t       wg_y_merge;
        ধ্রুবক uint32_t যোগ_wg_merge = nm * nrows;
        compute_2d_workgroups(যোগ_wg_merge, max_wg_per_dim, wg_x_merge, wg_y_merge);

        dispatches.push_back({
            argsort_merge_pipeline, std::move(merge_params), std::move(merge_entries), { wg_x_merge, wg_y_merge }
        });

        len <<= 1;
        in_is_tmp = !in_is_tmp;
    }

    ফেরত ggml_backend_webgpu_build_multi(ctx, dispatches);
}

স্থির webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = nullptr;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        যোগ_wg = ggml_nrows(dst);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির webgpu_encoded_op ggml_webgpu_যোগ_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    যৌক্তিক                  যোগ_যোগ = dst->op == GGML_OP_যোগ;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),
                                     যোগ_যোগ ? 0 : (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)),
                                     যোগ_যোগ ? 0 : (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)),
                                     যোগ_যোগ ? 0 : (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)),
                                     যোগ_যোগ ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     যোগ_যোগ ? 1 : (uint32_t) src->ne[1],
                                     যোগ_যোগ ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_যোগ_rows_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = যোগ_যোগ ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

স্থির যৌক্তিক ggml_webgpu_can_fuse_rms_norm_mul(ধ্রুবক কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx) {
    যদি (!ggml_can_fuse(cgraph, গ্রন্থি_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        ফেরত মিথ্যা;
    }

    // additional constraints specific to this fusion
    ধ্রুবক ggml_tensor * rms_norm = cgraph->nodes[গ্রন্থি_idx];
    ধ্রুবক ggml_tensor * mul      = cgraph->nodes[গ্রন্থি_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    যদি (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        ফেরত মিথ্যা;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    যদি (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        ফেরত মিথ্যা;
    }
    // rms_norm shader assumes contiguous rows
    যদি (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        ফেরত মিথ্যা;
    }

    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_webgpu_can_fuse_gdn_cache(ধ্রুবক কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, পূর্ণসংখ্যা & অঙ্ক_encoded_ops) {
    ধ্রুবক ggml_tensor * gdn = cgraph->nodes[গ্রন্থি_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    যদি (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক ggml_tensor * src_v     = gdn->src[2];
    ধ্রুবক int64_t       S_v       = src_v->ne[0];
    ধ্রুবক int64_t       H         = src_v->ne[1];
    ধ্রুবক int64_t       n_tokens  = src_v->ne[2];
    ধ্রুবক int64_t       n_seqs    = src_v->ne[3];
    ধ্রুবক int64_t       D         = S_v * S_v * H;
    ধ্রুবক int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    ধ্রুবক int64_t       n_written = std::min<int64_t>(n_tokens, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ধ্রুবক size_t লেজ_off = ggml_row_আকার(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ধ্রুবক ggml_tensor * cpy     = nullptr;
    পূর্ণসংখ্যা                 cpy_idx = 0;
    জন্য (পূর্ণসংখ্যা j = গ্রন্থি_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        ধ্রুবক ggml_tensor * n = cgraph->nodes[j];
        যদি (ggml_op_is_empty(n->op) || ggml_is_empty(n)) {
            চালু;
        }
        যদি (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            ফেরত মিথ্যা;
        }
        cpy     = n;
        cpy_idx = j;
    }
    যদি (cpy == nullptr) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ধ্রুবক ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    যদি (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != লেজ_off ||
        !ggml_is_contiguous(cpy_src)) {
        ফেরত মিথ্যা;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ধ্রুবক std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    যদি (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->তথ্য == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_আকার(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_row_আকার(GGML_TYPE_F32, D)) {
        ফেরত মিথ্যা;
    }

    অঙ্ক_encoded_ops = cpy_idx - গ্রন্থি_idx + 1;

    ফেরত সত্য;
}

স্থির webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ধ্রুবক uint32_t        mode_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_আকার(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_আকার(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_আকার(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_আকার(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_আকার(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_আকার(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_আকার(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_আকার(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_আকার(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_আকার(dst->type)),

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
    shader_lib_ctx.max_wg_আকার = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t যোগ_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_আকার);
    compute_2d_workgroups(যোগ_wg, ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);

    ফেরত ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
স্থির std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           পূর্ণসংখ্যা            গ্রন্থি_idx,
                                                           পূর্ণসংখ্যা &          অঙ্ক_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  গ্রন্থি  = nodes[গ্রন্থি_idx];

    যদি (ggml_is_empty(গ্রন্থি)) {
        ফেরত std::nullopt;
    }
    যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ফেরত std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << গ্রন্থি << ", " << ggml_op_name(গ্রন্থি->op) << ")");

    ggml_tensor * src0 = গ্রন্থি->src[0];
    ggml_tensor * src1 = গ্রন্থি->src[1];
    ggml_tensor * src2 = গ্রন্থি->src[2];

    নির্বাচন (গ্রন্থি->op) {
            // no-ops
        ক্ষেত্র GGML_OP_NONE:
        ক্ষেত্র GGML_OP_VIEW:
        ক্ষেত্র GGML_OP_PERMUTE:
        ক্ষেত্র GGML_OP_TRANSPOSE:
        ক্ষেত্র GGML_OP_RESHAPE:
            ফেরত std::nullopt;
        ক্ষেত্র GGML_OP_DUP:
        ক্ষেত্র GGML_OP_CPY:
        ক্ষেত্র GGML_OP_CONT:
            ফেরত ggml_webgpu_cpy(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_SET:
            ফেরত ggml_webgpu_set(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_SET_ROWS:
            ফেরত ggml_webgpu_set_rows(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_GET_ROWS:
            ফেরত ggml_webgpu_get_rows(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_MUL_MAT:
            ফেরত ggml_webgpu_mul_mat(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            ফেরত ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, গ্রন্থি);
        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            ফেরত ggml_webgpu_flash_attn(ctx, src0, src1, src2, গ্রন্থি->src[3], গ্রন্থি->src[4], গ্রন্থি);
        ক্ষেত্র GGML_OP_ADD:
        ক্ষেত্র GGML_OP_SUB:
        ক্ষেত্র GGML_OP_MUL:
        ক্ষেত্র GGML_OP_DIV:
            ফেরত ggml_webgpu_দ্বিমিক_op(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_ADD_ID:
            ফেরত ggml_webgpu_add_id(ctx, src0, src1, src2, গ্রন্থি);
        ক্ষেত্র GGML_OP_CONCAT:
            ফেরত ggml_webgpu_concat(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_REPEAT:
            ফেরত ggml_webgpu_repeat(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_RMS_NORM:
            যদি (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, গ্রন্থি_idx)) {
                অঙ্ক_encoded_ops        = 2;
                ggml_tensor * mul_গ্রন্থি = nodes[গ্রন্থি_idx + 1];
                ফেরত ggml_webgpu_rms_norm_mul(ctx, src0, গ্রন্থি, mul_গ্রন্থি->src[0], mul_গ্রন্থি->src[1], mul_গ্রন্থি);
            } অন্যথা {
                ফেরত ggml_webgpu_row_norm(ctx, src0, গ্রন্থি);
            }
        ক্ষেত্র GGML_OP_NORM:
        ক্ষেত্র GGML_OP_L2_NORM:
            ফেরত ggml_webgpu_row_norm(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_ROPE:
            ফেরত ggml_webgpu_rope(ctx, src0, src1, src2, গ্রন্থি);
        ক্ষেত্র GGML_OP_GLU:
            ফেরত ggml_webgpu_glu(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_SCALE:
            ফেরত ggml_webgpu_scale(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_SOFT_MAX:
            ফেরত ggml_webgpu_soft_max(ctx, src0, src1, src2, গ্রন্থি);
        ক্ষেত্র GGML_OP_UNARY:
        ক্ষেত্র GGML_OP_CLAMP:
        ক্ষেত্র GGML_OP_FILL:
        ক্ষেত্র GGML_OP_LOG:
        ক্ষেত্র GGML_OP_SQR:
        ক্ষেত্র GGML_OP_SQRT:
        ক্ষেত্র GGML_OP_SIN:
        ক্ষেত্র GGML_OP_COS:
        ক্ষেত্র GGML_OP_DIAG:
        ক্ষেত্র GGML_OP_TRI:
            ফেরত ggml_webgpu_unary_op(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_SOLVE_TRI:
            ফেরত ggml_webgpu_solve_tri(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_SSM_CONV:
            ফেরত ggml_webgpu_ssm_conv(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_SSM_SCAN:
            ফেরত ggml_webgpu_ssm_scan(ctx, src0, src1, src2, গ্রন্থি->src[3], গ্রন্থি->src[4], গ্রন্থি->src[5], গ্রন্থি->src[6],
                                        গ্রন্থি);
        ক্ষেত্র GGML_OP_GATED_DELTA_NET:
            যদি (ggml_webgpu_can_fuse_gdn_cache(cgraph, গ্রন্থি_idx, অঙ্ক_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[গ্রন্থি_idx + অঙ্ক_encoded_ops - 1]->src[1];
                ফেরত ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, গ্রন্থি->src[3], গ্রন্থি->src[4], গ্রন্থি->src[5],
                                                   গ্রন্থি, dst_fuse);
            } অন্যথা {
                ফেরত ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, গ্রন্থি->src[3], গ্রন্থি->src[4], গ্রন্থি->src[5],
                                                   গ্রন্থি, nullptr);
            }
        ক্ষেত্র GGML_OP_PAD:
            ফেরত ggml_webgpu_pad(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_ARGMAX:
            ফেরত ggml_webgpu_argmax(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_ARGSORT:
        ক্ষেত্র GGML_OP_শীর্ষ_K:
            // we reuse the same argsort implementation for top_k
            ফেরত ggml_webgpu_argsort(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_CUMSUM:
            ফেরত ggml_webgpu_cumsum(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_যোগ:
        ক্ষেত্র GGML_OP_যোগ_ROWS:
            ফেরত ggml_webgpu_যোগ_rows(ctx, src0, গ্রন্থি);
        ক্ষেত্র GGML_OP_CONV_2D:
            ফেরত ggml_webgpu_conv_2d(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_CONV_2D_DW:
            ফেরত ggml_webgpu_conv_2d_dw(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_IM2COL:
            ফেরত ggml_webgpu_im2col(ctx, src0, src1, গ্রন্থি);
        ক্ষেত্র GGML_OP_UPSCALE:
            ফেরত ggml_webgpu_upscale(ctx, src0, গ্রন্থি);
        পূর্বনির্ধারিত:
            ফেরত std::nullopt;
    }
}

#ifdef GGML_WEBGPU_GPU_PROFILE
স্থির শূন্য ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ধ্রুবক std::vector<std::string> & pipeline_names,
                                                        uint32_t &                       অঙ্ক_inflight_batches) {
    যদি (pipeline_names.empty()) {
        ফেরত;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.ResolveQuerySet(ctx->profile_timestamp_query_set, 0, ctx->profile_timestamp_query_গুনতি,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_বাফার_To_বাফার(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_গুনতি * আকার(uint64_t));

    wgpu::Command_বাফার profile_commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, অঙ্ক_inflight_batches);

    ধ্রুবক size_t mapped_আকার = ctx->profile_timestamp_query_গুনতি * আকার(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_গুনতি == 2 * pipeline_names.size());

    ggml_backend_webgpu_map_বাফার(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,
                                   mapped_আকার);
    ধ্রুবক uint64_t * ts_তথ্য = (ধ্রুবক uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_আকার);

    জন্য (size_t i = 0; i < pipeline_names.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ধ্রুবক দ্বিগুণ elapsed_ms = দ্বিগুণ(ts_তথ্য[2 * i + 1] - ts_তথ্য[2 * i]) * 1e-6;
        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#endif

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
স্থির শূন্য ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & অঙ্ক_inflight_batches) {
#ifdef GGML_WEBGPU_CHECK_SET_ROWS
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_বাফার_To_বাফার(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                               ctx->set_rows_host_error_buf.Get_আকার());
    wgpu::Command_বাফার commands = encoder.Finish();
    ggml_backend_webgpu_submit_commands(ctx, commands, অঙ্ক_inflight_batches);
    ggml_backend_webgpu_map_বাফার(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                   ctx->set_rows_host_error_buf.Get_আকার());
    ধ্রুবক uint32_t * error_তথ্য = (ধ্রুবক uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
    যদি (*error_তথ্য) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->set_rows_host_error_buf.Unmap();
#অন্যথা
    GGML_UNUSED(ctx);
    GGML_UNUSED(অঙ্ক_inflight_batches);
#endif
}

স্থির ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, কাঠামো ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_যোগ_START(graph_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t অঙ্ক_batched_kernels  = 0;
    uint32_t অঙ্ক_inflight_batches = 0;
    যৌক্তিক     contains_set_rows    = মিথ্যা;
    পূর্ণসংখ্যা      অঙ্ক_encoded_ops      = 1;
    পূর্ণসংখ্যা      গ্রন্থি_idx             = 0;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_গুনতি = 0;
    std::vector<std::string> profile_pipeline_names;
#endif

    ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    যদি (ctx->batch_compute_passes) {
        ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
    }

    যখন (গ্রন্থি_idx < cgraph->n_nodes) {
        যদি (cgraph->nodes[গ্রন্থি_idx]->op == GGML_OP_SET_ROWS) {
            contains_set_rows = সত্য;
        }
        যদি (auto cmd = ggml_webgpu_encode(ctx, cgraph, গ্রন্থি_idx, অঙ্ক_encoded_ops)) {
            commands.push_back(*cmd);
            অঙ্ক_batched_kernels += cmd.value().অঙ্ক_kernels;
#ifdef GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_names.insert(profile_pipeline_names.end(), cmd->pipeline_names.begin(),
                                          cmd->pipeline_names.end());
#endif
        }

        যদি (অঙ্ক_batched_kernels >= ctx->global_ctx->command_submit_batch_আকার) {
            যদি (ctx->active_compute_pass) {
                ctx->active_compute_pass.End();
            }
            অঙ্ক_batched_kernels                = 0;
            wgpu::Command_বাফার batch_commands = ctx->active_command_encoder.Finish();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, অঙ্ক_inflight_batches);

            // reset state for next batch
            ctx->active_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            যদি (ctx->batch_compute_passes) {
                ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#ifdef GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            যদি (ctx->profile_timestamp_query_গুনতি + 2 * ctx->global_ctx->command_submit_batch_আকার >=
                WEBGPU_MAX_PROFILE_QUERY_গুনতি) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, অঙ্ক_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_গুনতি = 0;
                profile_pipeline_names.clear();
            }
#endif
        }

        গ্রন্থি_idx += অঙ্ক_encoded_ops;
        অঙ্ক_encoded_ops = 1;
    }

    যদি (ctx->active_compute_pass) {
        ctx->active_compute_pass.End();
        ctx->active_compute_pass = nullptr;
    }

    যদি (অঙ্ক_batched_kernels > 0) {
        wgpu::Command_বাফার batch_commands = ctx->active_command_encoder.Finish();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, অঙ্ক_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->active_command_encoder = nullptr;

#ifdef GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_names, অঙ্ক_inflight_batches);
#endif

    যদি (contains_set_rows) {
        ggml_backend_webgpu_check_set_rows(ctx, অঙ্ক_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_যোগ_END(graph_compute, ctx->global_ctx);
    ফেরত GGML_STATUS_SUCCESS;
}

কাঠামো ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    যৌক্তিক                  recorded = মিথ্যা;
};

স্থির ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    auto * event_ctx      = নতুন ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    auto * event   = নতুন ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    ফেরত event;
}

স্থির শূন্য ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete static_cast<ggml_backend_webgpu_event_context *>(event->context);
    delete event;
}

স্থির শূন্য ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    যদি (!event_ctx->recorded) {
        ফেরত;
    }
    wgpu::WaitStatus status =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    যদি (status == wgpu::WaitStatus::TimedOut) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = মিথ্যা;
}

স্থির শূন্য ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
    event_ctx->recorded = সত্য;
}

স্থির শূন্য ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
}

স্থির শূন্য ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ধ্রুবক শূন্য *   তথ্য,
                                                 size_t         offset,
                                                 size_t         size) {
    GGML_UNUSED(backend);
    auto * buf_ctx      = (ggml_backend_webgpu_বাফার_context *) tensor->বাফার->context;
    size_t যোগ_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_বাফার(buf_ctx->বাফার, যোগ_offset, তথ্য, (size / 4) * 4);

    যদি (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_আকার = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        জন্য (size_t i = 0; i < remaining_আকার; i++) {
            ((uint8_t *) &val32)[i] = ((ধ্রুবক uint8_t *) তথ্য)[size - remaining_আকার + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_বাফার_memset(buf_ctx->global_ctx, buf_ctx->বাফার, val32,
                                          যোগ_offset + (size - remaining_আকার), remaining_আকার);
    }
}

স্থির শূন্য ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
}

স্থির ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
    /* .get_tensor_async        = */ ফাঁকা,
    /* .set_tensor_2d_async     = */ ফাঁকা,
    /* .get_tensor_2d_async     = */ ফাঁকা,
    /* .cpy_tensor_async        = */ ফাঁকা,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ ফাঁকা,
    /* .graph_plan_free         = */ ফাঁকা,
    /* .graph_plan_update       = */ ফাঁকা,
    /* .graph_plan_compute      = */ ফাঁকা,
    /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_record,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ ফাঁকা,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

স্থির শূন্য ggml_backend_webgpu_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    ggml_backend_webgpu_বাফার_context * ctx = static_cast<ggml_backend_webgpu_বাফার_context *>(বাফার->context);
    যদি (ctx != nullptr && ctx->বাফার != nullptr) {
        ctx->বাফার.Destroy();
        delete ctx;
    }
}

// Returns the "fake" base pointer.
স্থির শূন্য * ggml_backend_webgpu_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    GGML_UNUSED(বাফার);
    ফেরত webgpu_ptr_base;
}

স্থির শূন্য ggml_backend_webgpu_বাফার_memset_tensor(ggml_backend_বাফার_t বাফার,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                offset,
                                                     size_t                size) {
    যদি (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        ফেরত;
    }

    WEBGPU_CPU_PROFILE_যোগ_START(memset_tensor);

    ggml_backend_webgpu_বাফার_context * buf_ctx = (ggml_backend_webgpu_বাফার_context *) বাফার->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << offset << ", " << size << ")");

    size_t যোগ_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_বাফার_memset(buf_ctx->global_ctx, buf_ctx->বাফার, val32, যোগ_offset, size);
    WEBGPU_CPU_PROFILE_যোগ_END(memset_tensor, buf_ctx->global_ctx);
}

স্থির শূন্য ggml_backend_webgpu_বাফার_set_tensor(ggml_backend_বাফার_t বাফার,
                                                  ggml_tensor *         tensor,
                                                  ধ্রুবক শূন্য *          তথ্য,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_যোগ_START(set_tensor);
    ggml_backend_webgpu_বাফার_context * buf_ctx = (ggml_backend_webgpu_বাফার_context *) বাফার->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << তথ্য
                                                              << ", " << offset << ", " << size << ")");

    size_t যোগ_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    যদি (যোগ_offset % 4 != 0) {
        size_t lane = যোগ_offset % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t শীর্ষ = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t শীর্ষ_val = 0;
        জন্য (size_t i = 0; i < শীর্ষ; i++) {
            ((uint8_t *) &শীর্ষ_val)[lane + i] = ((ধ্রুবক uint8_t *) তথ্য)[i];
        }
        ggml_backend_webgpu_বাফার_memset(buf_ctx->global_ctx, buf_ctx->বাফার, শীর্ষ_val, যোগ_offset, শীর্ষ);

        যোগ_offset += শীর্ষ;
        size -= শীর্ষ;
        তথ্য = (ধ্রুবক uint8_t *) তথ্য + শীর্ষ;
    }

    যদি (size > 0) {
        buf_ctx->global_ctx->queue.Write_বাফার(buf_ctx->বাফার, যোগ_offset, তথ্য, (size / 4) * 4);
    }

    যদি (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t remaining_আকার = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        জন্য (size_t i = 0; i < remaining_আকার; i++) {
            ((uint8_t *) &val32)[i] = ((ধ্রুবক uint8_t *) তথ্য)[size - remaining_আকার + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_বাফার_memset(buf_ctx->global_ctx, buf_ctx->বাফার, val32,
                                          যোগ_offset + (size - remaining_আকার), remaining_আকার);
    }
    WEBGPU_CPU_PROFILE_যোগ_END(set_tensor, buf_ctx->global_ctx);
}

স্থির শূন্য ggml_backend_webgpu_বাফার_get_tensor(ggml_backend_বাফার_t বাফার,
                                                  ধ্রুবক ggml_tensor *   tensor,
                                                  শূন্য *                তথ্য,
                                                  size_t                offset,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_যোগ_START(get_tensor);
    ggml_backend_webgpu_বাফার_context * buf_ctx = (ggml_backend_webgpu_বাফার_context *) বাফার->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << তথ্য
                                                              << ", " << offset << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t যোগ_offset = ggml_webgpu_tensor_offset(tensor) + offset;

    size_t local_offset = যোগ_offset % 4;
    যদি (local_offset != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        যোগ_offset -= local_offset;
    }

    size_t final_আকার = size + local_offset;
    যদি (final_আকার % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_আকার += 4 - (final_আকার % 4);
    }

    std::lock_guard<std::recursive_mutex> lock(buf_ctx->global_ctx->mutex);

    যদি (buf_ctx->global_ctx->get_tensor_staging_buf == nullptr ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_আকার() < final_আকার) {
        // Create a new staging buffer if it doesn't exist or is too small
        যদি (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_বাফার(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_আকার,
                                  wgpu::বাফার_Usage::CopyDst | wgpu::বাফার_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_বাফার_To_বাফার(buf_ctx->বাফার, যোগ_offset, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_আকার);
    wgpu::Command_বাফার commands = encoder.Finish();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_বাফার(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::MapMode::Read, 0, final_আকার);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ধ্রুবক শূন্য * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_আকার);

    // Copy the data from the mapped range to the output buffer
    std::memcpy(তথ্য, (ধ্রুবক শূন্য *) ((ধ্রুবক অক্ষর *) mapped_range + local_offset), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_যোগ_END(get_tensor, buf_ctx->global_ctx);
}

স্থির শূন্য ggml_backend_webgpu_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << বাফার << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_যোগ_START(clear);
    ggml_backend_webgpu_বাফার_context * buf_ctx = (ggml_backend_webgpu_বাফার_context *) বাফার->context;
    ggml_backend_webgpu_বাফার_memset(buf_ctx->global_ctx, buf_ctx->বাফার, value, 0, বাফার->size);
    WEBGPU_CPU_PROFILE_যোগ_END(clear, buf_ctx->global_ctx);
}

স্থির ggml_backend_বাফার_i ggml_backend_webgpu_বাফার_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_webgpu_বাফার_get_base,
    /* .init_tensor     = */ ফাঁকা,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_বাফার_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_বাফার_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_বাফার_get_tensor,
    /* .set_tensor_2d   = */ ফাঁকা,
    /* .get_tensor_2d   = */ ফাঁকা,
    /* .cpy_tensor      = */ ফাঁকা,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_বাফার_clear,
    /* .reset           = */ ফাঁকা,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

স্থির ধ্রুবক অক্ষর * ggml_backend_webgpu_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ফেরত ctx->device_name.c_str();
}

স্থির ggml_backend_বাফার_t ggml_backend_webgpu_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft,
                                                                          size_t                     size) {
    স্থির std::atomic<পূর্ণসংখ্যা> বাফার_গুনতি;
    পূর্ণসংখ্যা                     বাফার_id = বাফার_গুনতি++;
    std::string             buf_name  = "tensor_buf" + std::to_string(বাফার_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << বাফার_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::বাফার_2                         buf;
    ggml_webgpu_create_বাফার(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::বাফার_Usage::Storage | wgpu::বাফার_Usage::CopySrc | wgpu::বাফার_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_বাফার_context * buf_ctx =
        নতুন ggml_backend_webgpu_বাফার_context(buf, buf_name, ctx->webgpu_global_ctx);

    ফেরত ggml_backend_বাফার_init(buft, ggml_backend_webgpu_বাফার_interface, buf_ctx, size);
}

স্থির size_t ggml_backend_webgpu_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ফেরত dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
স্থির size_t ggml_backend_webgpu_বাফার_type_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    ফেরত dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার;
}

স্থির size_t ggml_backend_webgpu_বাফার_type_get_alloc_আকার(ggml_backend_বাফার_type_t buft,
                                                             ধ্রুবক ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    নির্বাচন (tensor->op) {
        ক্ষেত্র GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            বিরতি;
        ক্ষেত্র GGML_OP_শীর্ষ_K:
            {
                ধ্রুবক ggml_tensor * src0 = tensor->src[0];
                যদি (src0) {
                    ধ্রুবক size_t full = আকার(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        full * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            {
                ধ্রুবক ggml_tensor * Q            = tensor->src[0];
                ধ্রুবক ggml_tensor * K            = tensor->src[1];
                ধ্রুবক ggml_tensor * V            = tensor->src[2];
                ধ্রুবক ggml_tensor * mask         = tensor->src[3];
                ধ্রুবক auto &        capabilities = ctx->webgpu_global_ctx->capabilities;
                যদি (ggml_webgpu_flash_attn_use_vec_path(ctx->webgpu_global_ctx, Q, K, V)) {
                    ধ্রুবক যৌক্তিক kv_direct =
                        ggml_webgpu_flash_attn_k_direct(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH) ||
                        ggml_webgpu_flash_attn_v_direct(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH);
                    ধ্রুবক uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_আকার, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        mask != nullptr, kv_direct);

                    ধ্রুবক uint32_t vec_nwg_cap = capabilities.min_subgroup_আকার;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ধ্রুবক size_t   align = capabilities.limits.min_Storage_বাফার_Offset_Alignment;
                    ধ্রুবক uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    যদি (nwg > 1u) {
                        ধ্রুবক uint64_t tmp_তথ্য_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ধ্রুবক uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ধ্রুবক size_t   tmp_আকার_bytes = ROUNDUP_POW2((tmp_তথ্য_elems + tmp_stats_elems) * আকার(দশমিক),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_আকার_bytes + align;
                    } অন্যথা {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    যদি (mask != nullptr) {
                        ধ্রুবক uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ধ্রুবক uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ধ্রুবক uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_আকার(mask->type));
                        ধ্রুবক uint32_t blk_batch_গুনতি = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ধ্রুবক uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_গুনতি;
                        ধ্রুবক size_t   blk_আকার_bytes =
                            ROUNDUP_POW2(blk_elems * আকার(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_আকার_bytes + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT:
            {
                ধ্রুবক ggml_tensor * src0 = tensor->src[0];
                ধ্রুবক ggml_tensor * src1 = tensor->src[1];
                যৌক্তিক                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_product,
                                             ctx->webgpu_global_ctx->vendor);
                যদি (use_mmvq) {
                    ধ্রুবক size_t q8_src1_আকার = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_আকার +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            {
                ধ্রুবক ggml_tensor * src0 = tensor->src[0];
                ধ্রুবক ggml_tensor * src1 = tensor->src[1];
                যদি (src0 && src1) {
                    ধ্রুবক size_t ids_cache_আকার = আকার(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    ধ্রুবক size_t ids_tiles_আকার = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_আকার + ids_tiles_আকার +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_বাফার_Offset_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            বিরতি;
        পূর্বনির্ধারিত:
            বিরতি;
    }
    ফেরত res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

স্থির ধ্রুবক অক্ষর * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    ফেরত ctx->device_name.c_str();
}

স্থির ধ্রুবক অক্ষর * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    ফেরত ctx->device_desc.c_str();
}

স্থির শূন্য ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_বাফার_আকার = ctx->webgpu_global_ctx->capabilities.limits.max_বাফার_আকার_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#যদি UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_আকার = static_cast<uint64_t>(UINTPTR_MAX);
    যদি (max_বাফার_আকার > max_ptr_আকার) {
        max_বাফার_আকার = max_ptr_আকার;
    }
#endif
    *মুক্ত  = static_cast<size_t>(max_বাফার_আকার);
    *যোগ = static_cast<size_t>(max_বাফার_আকার);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    ফেরত GGML_BACKEND_DEVICE_TYPE_GPU;
}

স্থির শূন্য ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, কাঠামো ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);
    props->caps = {
        /* .async                 = */ মিথ্যা,
        /* .host_buffer           = */ মিথ্যা,
        /* .buffer_from_host_ptr  = */ মিথ্যা,
        /* .events                = */ মিথ্যা,
        /* .mmap_support          = */ সত্য,
    };
}

স্থির ggml_guid_t ggml_backend_webgpu_guid(শূন্য) {
    স্থির ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    ফেরত &guid;
}

স্থির শূন্য ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_threads = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup *
                         ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_bytes_per_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার, max_threads);
    std::vector<wgpu::ConstantEntry> constants(2);
    constants[0].চাবি     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
    constants[1].চাবি     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

স্থির শূন্য ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#যদি !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ধ্রুবক অক্ষর * ধ্রুবক          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_গুনতি = 2;
    options.পরবর্তী_In_Chain                   = &adapterTogglesDesc;
#endif

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter _adapter, wgpu::StringView message) {
                যদি (status != wgpu::RequestAdapterStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(message).c_str());
                    ফেরত;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

স্থির শূন্য create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != nullptr);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#যদি !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
    যদি (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        info.পরবর্তী_In_Chain = &subgroup_matrix_configs;
    }
#endif
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_আকার = ggml_backend_webgpu_get_command_submit_batch_আকার();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_product = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8IntegerDotProduct);

    যৌক্তিক valid_subgroup_matrix_config = মিথ্যা;
#যদি !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    যদি (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        জন্য (size_t i = 0; i < subgroup_matrix_configs.config_গুনতি; i++) {
            ধ্রুবক wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
            যদি (config.componentType == wgpu::SubgroupMatrixComponentType::F16 &&
                config.ফলাফল_Component_Type == wgpu::SubgroupMatrixComponentType::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_matrix_config                  = সত্য;
                বিরতি;
            }
        }
    }
#endif
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_আকার = info.subgroup_Min_আকার;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_আকার = info.subgroup_Max_আকার;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#যদি !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    যদি (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
        required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
    }
#endif

    যদি (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#ifdef GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#endif

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.তথ্য();
    dev_desc.required_Feature_গুনতি = required_features.size();
    dev_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](ধ্রুবক wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            যদি (reason == wgpu::DeviceLostReason::Destroyed) {
                ফেরত;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<পূর্ণসংখ্যা>(reason),
                           std::string(message).c_str());
        });
    dev_desc.SetUncapturedErrorCallback(
        [](ধ্রুবক wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<পূর্ণসংখ্যা>(reason),
                       std::string(message).c_str());
        });

#যদি !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    ধ্রুবক অক্ষর * ধ্রুবক          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    ধ্রুবক অক্ষর * ধ্রুবক          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_গুনতি  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_গুনতি = 1;

    dev_desc.পরবর্তী_In_Chain = &deviceTogglesDesc;
#endif

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::CallbackMode::AllowSpontaneous,
            [ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                যদি (status != wgpu::RequestDeviceStatus::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(message).c_str());
                    ফেরত;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_বাফার(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_আকার_BYTES, wgpu::বাফার_Usage::CopyDst | wgpu::বাফার_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.GetQueue();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

স্থির webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_আকার_BYTES,
        webgpu_ctx->global_ctx->command_submit_batch_আকার + WEBGPU_অঙ্ক_PARAM_SLOT_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_Uniform_বাফার_Offset_Alignment);
    ggml_webgpu_create_বাফার(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_আকার_BYTES,
                              wgpu::বাফার_Usage::Storage | wgpu::বাফার_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_বাফার(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_host_error_buf,
                              WEBGPU_SET_ROWS_ERROR_BUF_আকার_BYTES,
                              wgpu::বাফার_Usage::CopyDst | wgpu::বাফার_Usage::MapRead, "set_rows_host_error_buf");

#ifdef GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = মিথ্যা;
    ggml_webgpu_create_বাফার(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_আকার_BYTES,
        wgpu::বাফার_Usage::QueryResolve | wgpu::বাফার_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_বাফার(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_আকার_BYTES,
                              wgpu::বাফার_Usage::CopyDst | wgpu::বাফার_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::QuerySetDescriptor query_set_desc = {};
    query_set_desc.type                     = wgpu::QueryType::Timestamp;
    query_set_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_গুনতি;
    webgpu_ctx->profile_timestamp_query_set = webgpu_ctx->global_ctx->device.CreateQuerySet(&query_set_desc);
#endif

#ifdef GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_বাফার(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * আকার(uint32_t),
                              wgpu::বাফার_Usage::CopyDst | wgpu::বাফার_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_বাফার(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * আকার(uint32_t),
                              wgpu::বাফার_Usage::Storage | wgpu::বাফার_Usage::CopySrc, "debug_dev_buf");
#endif
    ফেরত webgpu_ctx;
}

স্থির ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    auto * backend_ctx      = নতুন ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    auto * backend = নতুন ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    ফেরত backend;
}

স্থির ggml_backend_বাফার_type_t ggml_backend_webgpu_device_get_বাফার_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    স্থির কাঠামো ggml_backend_বাফার_type ggml_backend_webgpu_বাফার_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_বাফার_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_বাফার_type_alloc_বাফার,
                        /* .alloc_buffer_n      = */ ফাঁকা,
                        /* .get_alignment       = */ ggml_backend_webgpu_বাফার_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_বাফার_type_get_max_আকার,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_বাফার_type_get_alloc_আকার,
                        /* .get_alloc_size_n    = */ ফাঁকা,
                        /* .is_host             = */ ফাঁকা,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ ফাঁকা
    };

    ফেরত &ggml_backend_webgpu_বাফার_type;
}

স্থির যৌক্তিক ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    GGML_UNUSED(dev);
    ফেরত buft->iface.get_name == ggml_backend_webgpu_বাফার_type_get_name;
}

স্থির যৌক্তিক ggml_webgpu_supported_qtype(ggml_type type) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_Q1_0:
        ক্ষেত্র GGML_TYPE_Q2_0:
        ক্ষেত্র GGML_TYPE_Q4_0:
        ক্ষেত্র GGML_TYPE_Q4_1:
        ক্ষেত্র GGML_TYPE_Q5_0:
        ক্ষেত্র GGML_TYPE_Q5_1:
        ক্ষেত্র GGML_TYPE_Q8_0:
        ক্ষেত্র GGML_TYPE_Q2_K:
        ক্ষেত্র GGML_TYPE_Q3_K:
        ক্ষেত্র GGML_TYPE_Q4_K:
        ক্ষেত্র GGML_TYPE_Q5_K:
        ক্ষেত্র GGML_TYPE_Q6_K:
        ক্ষেত্র GGML_TYPE_IQ2_XXS:
        ক্ষেত্র GGML_TYPE_IQ2_XS:
        ক্ষেত্র GGML_TYPE_IQ2_S:
        ক্ষেত্র GGML_TYPE_IQ3_XXS:
        ক্ষেত্র GGML_TYPE_IQ3_S:
        ক্ষেত্র GGML_TYPE_IQ1_S:
        ক্ষেত্র GGML_TYPE_IQ1_M:
        ক্ষেত্র GGML_TYPE_IQ4_NL:
        ক্ষেত্র GGML_TYPE_IQ4_XS:
        ক্ষেত্র GGML_TYPE_MXFP4:
        ক্ষেত্র GGML_TYPE_NVFP4:
            ফেরত সত্য;
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }
}

স্থির যৌক্তিক ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    যদি (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার)) {
        ফেরত মিথ্যা;
    }

    যৌক্তিক supports_op = মিথ্যা;
    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_NONE:
        ক্ষেত্র GGML_OP_VIEW:
        ক্ষেত্র GGML_OP_PERMUTE:
        ক্ষেত্র GGML_OP_TRANSPOSE:
        ক্ষেত্র GGML_OP_RESHAPE:
            supports_op = সত্য;
            বিরতি;
        ক্ষেত্র GGML_OP_ADD:
        ক্ষেত্র GGML_OP_SUB:
        ক্ষেত্র GGML_OP_MUL:
        ক্ষেত্র GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_ADD_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            বিরতি;
        ক্ষেত্র GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            বিরতি;
        ক্ষেত্র GGML_OP_DUP:
        ক্ষেত্র GGML_OP_CPY:
        ক্ষেত্র GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            বিরতি;
        ক্ষেত্র GGML_OP_SET:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            বিরতি;
        ক্ষেত্র GGML_OP_SET_ROWS:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            বিরতি;
        ক্ষেত্র GGML_OP_GET_ROWS:
            যদি (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } অন্যথা যদি (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT:
            {
                নির্বাচন (src1->type) {
                    ক্ষেত্র GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        বিরতি;
                    ক্ষেত্র GGML_TYPE_F32:
                        নির্বাচন (src0->type) {
                            ক্ষেত্র GGML_TYPE_F32:
                            ক্ষেত্র GGML_TYPE_F16:
                            ক্ষেত্র GGML_TYPE_BF16:
                            ক্ষেত্র GGML_TYPE_Q1_0:
                            ক্ষেত্র GGML_TYPE_Q2_0:
                            ক্ষেত্র GGML_TYPE_Q4_0:
                            ক্ষেত্র GGML_TYPE_Q4_1:
                            ক্ষেত্র GGML_TYPE_Q5_0:
                            ক্ষেত্র GGML_TYPE_Q5_1:
                            ক্ষেত্র GGML_TYPE_Q8_0:
                            ক্ষেত্র GGML_TYPE_Q2_K:
                            ক্ষেত্র GGML_TYPE_Q3_K:
                            ক্ষেত্র GGML_TYPE_Q4_K:
                            ক্ষেত্র GGML_TYPE_Q5_K:
                            ক্ষেত্র GGML_TYPE_Q6_K:
                            ক্ষেত্র GGML_TYPE_IQ2_XXS:
                            ক্ষেত্র GGML_TYPE_IQ2_XS:
                            ক্ষেত্র GGML_TYPE_IQ2_S:
                            ক্ষেত্র GGML_TYPE_IQ3_XXS:
                            ক্ষেত্র GGML_TYPE_IQ3_S:
                            ক্ষেত্র GGML_TYPE_IQ1_S:
                            ক্ষেত্র GGML_TYPE_IQ1_M:
                            ক্ষেত্র GGML_TYPE_IQ4_NL:
                            ক্ষেত্র GGML_TYPE_IQ4_XS:
                            ক্ষেত্র GGML_TYPE_MXFP4:
                            ক্ষেত্র GGML_TYPE_NVFP4:
                                supports_op = সত্য;
                                বিরতি;
                            পূর্বনির্ধারিত:
                                বিরতি;
                        }
                    পূর্বনির্ধারিত:
                        বিরতি;
                }
                বিরতি;
            }
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            নির্বাচন (src1->type) {
                ক্ষেত্র GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    বিরতি;
                ক্ষেত্র GGML_TYPE_F32:
                    নির্বাচন (src0->type) {
                        ক্ষেত্র GGML_TYPE_F32:
                        ক্ষেত্র GGML_TYPE_F16:
                        ক্ষেত্র GGML_TYPE_BF16:
                        ক্ষেত্র GGML_TYPE_Q1_0:
                        ক্ষেত্র GGML_TYPE_Q4_0:
                        ক্ষেত্র GGML_TYPE_Q4_1:
                        ক্ষেত্র GGML_TYPE_Q5_0:
                        ক্ষেত্র GGML_TYPE_Q5_1:
                        ক্ষেত্র GGML_TYPE_Q8_0:
                        ক্ষেত্র GGML_TYPE_Q2_K:
                        ক্ষেত্র GGML_TYPE_Q3_K:
                        ক্ষেত্র GGML_TYPE_Q4_K:
                        ক্ষেত্র GGML_TYPE_Q5_K:
                        ক্ষেত্র GGML_TYPE_Q6_K:
                        ক্ষেত্র GGML_TYPE_IQ1_S:
                        ক্ষেত্র GGML_TYPE_IQ1_M:
                        ক্ষেত্র GGML_TYPE_IQ2_XXS:
                        ক্ষেত্র GGML_TYPE_IQ2_XS:
                        ক্ষেত্র GGML_TYPE_IQ2_S:
                        ক্ষেত্র GGML_TYPE_IQ3_XXS:
                        ক্ষেত্র GGML_TYPE_IQ3_S:
                        ক্ষেত্র GGML_TYPE_IQ4_NL:
                        ক্ষেত্র GGML_TYPE_IQ4_XS:
                        ক্ষেত্র GGML_TYPE_MXFP4:
                        ক্ষেত্র GGML_TYPE_NVFP4:
                            supports_op = সত্য;
                            বিরতি;
                        পূর্বনির্ধারিত:
                            বিরতি;
                    }
                    বিরতি;
                পূর্বনির্ধারিত:
                    বিরতি;
            }
            যদি (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                যদি (!supports_op) {
                    বিরতি;
                }
                যদি (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = মিথ্যা;
                    বিরতি;
                }
                ধ্রুবক auto & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ধ্রুবক size_t storage_offset_alignment = capabilities.limits.min_Storage_বাফার_Offset_Alignment;

                // subgroup matrix path requirements
                ধ্রুবক যৌক্তিক use_subgroup_matrix = ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(
                    capabilities.supports_subgroup_matrix, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ধ্রুবক যৌক্তিক float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_offset_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_offset_alignment));
                ধ্রুবক uint32_t k_tile_শীর্ষ_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_আকার(src1->type);
                ধ্রুবক uint32_t v_tile_শীর্ষ_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH :
                                                       (uint32_t) ggml_blck_আকার(src2->type);
                ধ্রুবক যৌক্তিক     tile_kv_শীর্ষ_dims_aligned =
                    src0->ne[0] % k_tile_শীর্ষ_align == 0 && src2->ne[0] % v_tile_শীর্ষ_align == 0;
                ধ্রুবক যৌক্তিক tile_can_dispatch_all_q_rows =
                    capabilities.limits.maxComputeInvocationsPerWorkgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_আকার;
                ধ্রুবক যৌক্তিক use_tile = !use_subgroup_matrix && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_শীর্ষ_dims_aligned && tile_can_dispatch_all_q_rows;

                যদি (!use_subgroup_matrix && !use_tile) {
                    supports_op = মিথ্যা;
                    বিরতি;
                }
                ধ্রুবক uint32_t q_tile =
                    use_subgroup_matrix ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ধ্রুবক uint32_t kv_granularity = use_subgroup_matrix ? capabilities.sg_mat_n : 1u;
                ধ্রুবক যৌক্তিক     kv_direct = use_subgroup_matrix ?
                                               ggml_webgpu_flash_attn_k_direct(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_direct(src0, src2, capabilities.sg_mat_k) :
                                               মিথ্যা;
                ধ্রুবক uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_আকার, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != nullptr, kv_direct);
                supports_op = max_kv_tile > 0;
                বিরতি;
            }
        ক্ষেত্র GGML_OP_RMS_NORM:
        ক্ষেত্র GGML_OP_NORM:
        ক্ষেত্র GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_rows(src0);
            বিরতি;
        ক্ষেত্র GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            বিরতি;
        ক্ষেত্র GGML_OP_GLU:
            নির্বাচন (ggml_get_glu_op(op)) {
                ক্ষেত্র GGML_GLU_OP_REGLU:
                ক্ষেত্র GGML_GLU_OP_GEGLU:
                ক্ষেত্র GGML_GLU_OP_SWIGLU:
                ক্ষেত্র GGML_GLU_OP_GEGLU_ERF:
                ক্ষেত্র GGML_GLU_OP_GEGLU_QUICK:
                ক্ষেত্র GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    বিরতি;
                ক্ষেত্র GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    বিরতি;
                পূর্বনির্ধারিত:
                    বিরতি;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_UNARY:
            {
                ধ্রুবক ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                নির্বাচন (UNARY_OP) {
                    ক্ষেত্র GGML_UNARY_OP_ABS:
                    ক্ষেত্র GGML_UNARY_OP_SGN:
                    ক্ষেত্র GGML_UNARY_OP_NEG:
                    ক্ষেত্র GGML_UNARY_OP_STEP:
                    ক্ষেত্র GGML_UNARY_OP_TANH:
                    ক্ষেত্র GGML_UNARY_OP_ELU:
                    ক্ষেত্র GGML_UNARY_OP_RELU:
                    ক্ষেত্র GGML_UNARY_OP_SIGMOID:
                    ক্ষেত্র GGML_UNARY_OP_GELU:
                    ক্ষেত্র GGML_UNARY_OP_GELU_QUICK:
                    ক্ষেত্র GGML_UNARY_OP_SILU:
                    ক্ষেত্র GGML_UNARY_OP_HARDSWISH:
                    ক্ষেত্র GGML_UNARY_OP_HARDSIGMOID:
                    ক্ষেত্র GGML_UNARY_OP_EXP:
                    ক্ষেত্র GGML_UNARY_OP_GELU_ERF:
                    ক্ষেত্র GGML_UNARY_OP_SOFTPLUS:
                    ক্ষেত্র GGML_UNARY_OP_EXPM1:
                    ক্ষেত্র GGML_UNARY_OP_FLOOR:
                    ক্ষেত্র GGML_UNARY_OP_CEIL:
                    ক্ষেত্র GGML_UNARY_OP_ROUND:
                    ক্ষেত্র GGML_UNARY_OP_TRUNC:
                    ক্ষেত্র GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        বিরতি;
                    পূর্বনির্ধারিত:
                        বিরতি;
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            বিরতি;
        ক্ষেত্র GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            বিরতি;
        ক্ষেত্র GGML_OP_IM2COL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            বিরতি;
        ক্ষেত্র GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_SSM_SCAN:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            বিরতি;
        ক্ষেত্র GGML_OP_GATED_DELTA_NET:
            {
                ধ্রুবক uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            বিরতি;
        ক্ষেত্র GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            বিরতি;
        ক্ষেত্র GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            বিরতি;
        ক্ষেত্র GGML_OP_শীর্ষ_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
            বিরতি;
        ক্ষেত্র GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            বিরতি;
        ক্ষেত্র GGML_OP_যোগ:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            বিরতি;
        ক্ষেত্র GGML_OP_যোগ_ROWS:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
            বিরতি;
        ক্ষেত্র GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            বিরতি;
        পূর্বনির্ধারিত:
            বিরতি;
    }
    যদি (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার ||
        (src0 != nullptr &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার) ||
        (src1 != nullptr &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার) ||
        (src2 != nullptr &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_বাফার_Binding_আকার)) {
        supports_op = মিথ্যা;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    যদি (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } অন্যথা {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    ফেরত supports_op;
}

স্থির কাঠামো ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_বাফার_type,
    /* .get_host_buffer_type = */ ফাঁকা,
    /* .buffer_from_host_ptr = */ ফাঁকা,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ ফাঁকা,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

স্থির ধ্রুবক অক্ষর * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    ফেরত ctx->name;
}

স্থির size_t ggml_backend_webgpu_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    ফেরত ctx->device_গুনতি;
}

// Only one device is supported for now
স্থির ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t সূচক) {
    GGML_ASSERT(সূচক == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_যোগ_START(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    স্থির ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    স্থির ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_যোগ_END(reg_get_device, reg_ctx->webgpu_global_ctx);
    ফেরত &device;
}

স্থির ধ্রুবক কাঠামো ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ ফাঁকা,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    স্থির ggml_backend_webgpu_reg_context * ctx = নতুন ggml_backend_webgpu_reg_context();

    স্থির ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_গুনতি = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    যদি (ctx->webgpu_global_ctx != nullptr && ctx->webgpu_global_ctx->instance != nullptr) {
        ফেরত &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.তথ্য();
    instance_descriptor.required_Feature_গুনতি                 = instance_features.size();

#যদি !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ধ্রুবক অক্ষর * ধ্রুবক          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_গুনতি = 1;
    instance_descriptor.পরবর্তী_In_Chain        = &instanceTogglesDesc;
#endif

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(নতুন webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    যদি (ctx->webgpu_global_ctx->instance != nullptr) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    যদি (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#যদি !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#endif
    ) {
        ctx->device_গুনতি = 1;
    }

    ফেরত &reg;
}

ggml_backend_t ggml_backend_webgpu_init(শূন্য) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    যদি (ggml_backend_reg_dev_গুনতি(reg) == 0) {
        ফেরত nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ফেরত ggml_backend_webgpu_backend_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
