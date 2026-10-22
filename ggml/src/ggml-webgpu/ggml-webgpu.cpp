/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#شامل "ggml-webgpu.h"

#شامل "ggml-backend-impl.h"
#شامل "ggml-impl.h"
#شامل "ggml-webgpu-shader-lib.hpp"
#شامل "ggml.h"

#اگر_تعریف_شدہ __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#اگر_ختم

#شامل <webgpu/webgpu_cpp.h>

#شامل <atomic>
#شامل <cstdint>
#شامل <cstring>
#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#اگر_ختم
#اگر defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#اگر_ختم
#شامل <memory>
#شامل <mutex>
#شامل <optional>
#شامل <string>
#شامل <utility>
#شامل <vector>

#تعریف_کرو ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#تعریف_کرو CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
ساکن اندرونی خالی compute_2d_workgroups(uint32_t کل_wg, uint32_t max_فی_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(کل_wg, max_فی_dim));
    wg_x = CEIL_DIV(کل_wg, wg_y);
}

ساکن اندرونی uint32_t ggml_webgpu_u32_from_f32(اعشاری value) {
    uint32_t بٹ;
    حافظہ_نقل(&بٹ, &value, حجم(بٹ));
    واپس بٹ;
}

#اگر_تعریف_شدہ GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(پیغام_2)  std::خارج << پیغام_2 << std::سطر_ختم
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#ورنہ
#    define WEBGPU_LOG_DEBUG(پیغام_2) ((خالی) 0)
#اگر_ختم  // GGML_WEBGPU_DEBUG

#اگر_تعریف_شدہ GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_کل_آغاز(id) خودکار cpu_کل_آغاز##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_کل_انتہا(id, ctx)                                                         \
        خودکار   cpu_کل_انتہا##id = std::chrono::high_resolution_clock::now();                            \
        دوہرا cpu_کل_وقت##id =                                                                      \
            std::chrono::duration<دوہرا, std::milli>(cpu_کل_انتہا##id - cpu_کل_آغاز##id).count(); \
        (ctx)->cpu_وقت_ms[#id] += cpu_کل_وقت##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_آغاز(id) خودکار cpu_detail_آغاز##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_انتہا(id, ctx)                                                          \
        خودکار   cpu_detail_انتہا##id = std::chrono::high_resolution_clock::now();                             \
        دوہرا cpu_detail_وقت##id =                                                                       \
            std::chrono::duration<دوہرا, std::milli>(cpu_detail_انتہا##id - cpu_detail_آغاز##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_وقت##id;
#ورنہ
#    define WEBGPU_CPU_PROFILE_کل_آغاز(id)
#    define WEBGPU_CPU_PROFILE_کل_انتہا(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_آغاز(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_انتہا(id, ctx)
#اگر_ختم  // GGML_WEBGPU_CPU_PROFILE

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_گنتی        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_سائز_بائٹیں (WEBGPU_MAX_PROFILE_QUERY_گنتی * حجم(uint64_t))
#اگر_ختم

/* Constants */

#تعریف_کرو WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_سائز 64u
#تعریف_کرو WEBGPU_عدد_PARAM_خانہ_SAFETY_MARGIN      10u
#تعریف_کرو WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#تعریف_کرو WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#تعریف_کرو WEBGPU_PARAMS_BUF_سائز_بائٹیں             128  // enough for 32 parameters
#تعریف_کرو WEBGPU_سیٹ_صفیں_ERROR_BUF_سائز_بائٹیں     4
#تعریف_کرو WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
ساکن خالی * ثابت webgpu_ptr_بنیاد = (خالی *) (uintptr_t) 0x1000;  // NOLINT

ساکن size_t ggml_webgpu_tensor_انحراف(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * بنیاد_tensor = tensor->view_src ? tensor->view_src : tensor;
    واپس (size_t) ((uintptr_t) بنیاد_tensor->ڈیٹا - (uintptr_t) webgpu_ptr_بنیاد) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
ساکن خالی ggml_webgpu_create_بفر(wgpu::Device &    device,
                                      wgpu::بفر_2 &    بفر,
                                      size_t            size,
                                      wgpu::بفر_Usage usage,
                                      ثابت حرف *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
ساخت webgpu_param_arena {
    wgpu::بفر_2 بفر;
    size_t       خانہ_stride = 0;
    size_t       خانہ_سائز   = 0;
    uint32_t     خانہ_گنتی  = 0;
    uint32_t     اگلا_خانہ   = 0;

    خالی init(wgpu::Device device, size_t خانہ_سائز, uint32_t خانہ_گنتی, size_t alignment) {
        یہ->خانہ_stride = ROUNDUP_POW2(خانہ_سائز, alignment);
        یہ->خانہ_سائز   = خانہ_سائز;
        یہ->خانہ_گنتی  = خانہ_گنتی;
        یہ->اگلا_خانہ   = 0;

        ggml_webgpu_create_بفر(device, بفر, یہ->خانہ_stride * خانہ_گنتی,
                                  wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_خانہ(size_t size) {
        GGML_ASSERT(size <= خانہ_سائز);
        اگر (اگلا_خانہ >= خانہ_گنتی) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        واپس خانہ_stride * اگلا_خانہ++;
    }

    خالی reset() { اگلا_خانہ = 0; }

    خالی cleanup() {
        اگر (بفر) {
            بفر.Destroy();
            بفر = خالی_اشاریہ;
        }
    }

    ~webgpu_param_arena() { یہ->cleanup(); }
};

ساخت webgpu_encoded_op {
    uint32_t عدد_kernels = 0;
#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_نام;
#اگر_ختم
};

ساخت webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_گروہ_اندراج> bind_گروہ_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

ساخت webgpu_capabilities {
    wgpu::Limits limits;
    منطقی         supports_subgroups       = جھوٹ;
    منطقی         supports_subgroup_میٹرکس = جھوٹ;
    منطقی         supports_dot_حاصل_ضرب     = جھوٹ;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_سائز     = 0;
    uint32_t min_subgroup_سائز = 0;
    uint32_t max_subgroup_سائز = 0;
    size_t   memset_بائٹیں_فی_thread;
};

// Stores global webgpu members
ساخت webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::قطار_2    queue;
    uint32_t       command_submit_batch_سائز = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_سائز;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::بفر_2         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::بفر_2    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#اگر_تعریف_شدہ GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, دوہرا> cpu_وقت_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, دوہرا> cpu_detail_ms;
#اگر_ختم

#اگر_تعریف_شدہ GGML_WEBGPU_DEBUG
    wgpu::بفر_2 debug_host_buf;
    wgpu::بفر_2 debug_dev_buf;
#اگر_ختم

    ~webgpu_global_context_struct() {
        اگر (یہ->get_tensor_staging_buf) {
            یہ->get_tensor_staging_buf.Destroy();
            یہ->get_tensor_staging_buf = خالی_اشاریہ;
        }
        اگر (یہ->memset_params_buf) {
            یہ->memset_params_buf.Destroy();
            یہ->memset_params_buf = خالی_اشاریہ;
        }
#اگر_تعریف_شدہ GGML_WEBGPU_DEBUG
        اگر (یہ->debug_host_buf) {
            یہ->debug_host_buf.Destroy();
            یہ->debug_host_buf = خالی_اشاریہ;
        }
        اگر (یہ->debug_dev_buf) {
            یہ->debug_dev_buf.Destroy();
            یہ->debug_dev_buf = خالی_اشاریہ;
        }
#اگر_ختم
    }
};

قسمتعریف std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
ساخت webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::بفر_2             سیٹ_صفیں_dev_error_buf;
    wgpu::بفر_2             سیٹ_صفیں_host_error_buf;
    wgpu::CommandEncoder     فعال_command_encoder;
    wgpu::ComputePassEncoder فعال_compute_pass;
    منطقی                     batch_compute_passes = سچ;

    size_t memset_بائٹیں_فی_thread;

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, دوہرا> shader_gpu_وقت_ms;
    wgpu::بفر_2                            profile_timestamp_dev_buf;
    wgpu::بفر_2                            profile_timestamp_host_buf;
    wgpu::Query_سیٹ                          profile_timestamp_query_سیٹ;
    uint32_t                                profile_timestamp_query_گنتی = 0;
#اگر_ختم

    ~webgpu_context_struct() {
#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
        اگر (یہ->profile_timestamp_host_buf) {
            یہ->profile_timestamp_host_buf.Destroy();
            یہ->profile_timestamp_host_buf = خالی_اشاریہ;
        }
        اگر (یہ->profile_timestamp_dev_buf) {
            یہ->profile_timestamp_dev_buf.Destroy();
            یہ->profile_timestamp_dev_buf = خالی_اشاریہ;
        }
        اگر (یہ->profile_timestamp_query_سیٹ) {
            یہ->profile_timestamp_query_سیٹ.Destroy();
            یہ->profile_timestamp_query_سیٹ = خالی_اشاریہ;
        }
#اگر_ختم
        اگر (یہ->سیٹ_صفیں_host_error_buf) {
            یہ->سیٹ_صفیں_host_error_buf.Destroy();
            یہ->سیٹ_صفیں_host_error_buf = خالی_اشاریہ;
        }
        اگر (یہ->سیٹ_صفیں_dev_error_buf) {
            یہ->سیٹ_صفیں_dev_error_buf.Destroy();
            یہ->سیٹ_صفیں_dev_error_buf = خالی_اشاریہ;
        }
    }
};

قسمتعریف std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
ساخت ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_گنتی;
    ثابت حرف *          name;
};

// Per-device struct for the global logical device interface
ساخت ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
ساخت ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
ساخت ggml_backend_webgpu_بفر_context {
    wgpu::بفر_2          بفر;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_بفر_context(wgpu::بفر_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        بفر(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

ساکن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ثابت حرف *                             shader_رمز,
                                                   ثابت حرف *                             label,
                                                   ثابت std::vector<wgpu::Constant_اندراج> & constants = {}) {
    wgpu::Shader_ماخذ_WGSL shader_ماخذ;
    shader_ماخذ.code = shader_رمز;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.اگلا_اندر_Chain = &shader_ماخذ;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.اندراج_نقطہ_2 = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = خالی_اشاریہ;  // nullptr means auto layout
    اگر (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.ڈیٹا();
        pipeline_desc.compute.constant_گنتی = constants.size();
    }
    واپس { device.CreateComputePipeline(&pipeline_desc), label };
}

ساکن خالی ggml_webgpu_create_بفر(wgpu::Device &    device,
                                      wgpu::بفر_2 &    بفر,
                                      size_t            size,
                                      wgpu::بفر_Usage usage,
                                      ثابت حرف *      label) {
    wgpu::بفر_Descriptor بفر_desc;
    بفر_desc.size             = size;
    بفر_desc.usage            = usage;
    بفر_desc.label            = label;
    بفر_desc.mappedAtCreation = جھوٹ;

    // TODO: error handling
    بفر = device.Create_بفر(&بفر_desc);
}

ساکن wgpu::بفر_2 ggml_webgpu_tensor_buf(ثابت ggml_tensor * tensor) {
    ggml_backend_webgpu_بفر_context * ctx = (ggml_backend_webgpu_بفر_context *) tensor->بفر->context;
    واپس ctx->بفر;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
ساکن size_t ggml_webgpu_tensor_align_انحراف(ثابت ggml_tensor * t, size_t alignment) {
    ثابت size_t انحراف    = ggml_webgpu_tensor_انحراف(t);
    ثابت size_t type_سائز = ggml_type_سائز(t->type);
    size_t       aligned   = انحراف & ~(alignment - 1);
    جب ((انحراف - aligned) % type_سائز != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    واپس aligned;
}

ساکن size_t ggml_webgpu_tensor_misalignment(ثابت ggml_tensor * t, size_t alignment) {
    واپس ggml_webgpu_tensor_انحراف(t) - ggml_webgpu_tensor_align_انحراف(t, alignment);
}

ساکن size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ثابت ggml_tensor * t) {
    واپس ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment);
}

ساکن size_t ggml_webgpu_tensor_align_انحراف(webgpu_context & ctx, ثابت ggml_tensor * t) {
    واپس ggml_webgpu_tensor_align_انحراف(t, ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment);
}

ساکن size_t ggml_webgpu_tensor_binding_سائز(ثابت ggml_tensor * t, size_t alignment) {
    واپس ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

ساکن size_t ggml_webgpu_tensor_binding_سائز(webgpu_context & ctx, ثابت ggml_tensor * t) {
    واپس ggml_webgpu_tensor_binding_سائز(t, ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment);
}

ساکن منطقی ggml_webgpu_tensor_binding_overlap(ثابت webgpu_global_context & global_ctx,
                                               ثابت ggml_tensor *           a,
                                               ثابت ggml_tensor *           b) {
    اگر (a->بفر != b->بفر) {
        واپس جھوٹ;
    }

    ثابت size_t alignment = global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment;
    ثابت size_t a_انحراف  = ggml_webgpu_tensor_align_انحراف(a, alignment);
    ثابت size_t b_انحراف  = ggml_webgpu_tensor_align_انحراف(b, alignment);
    واپس a_انحراف < b_انحراف + ggml_webgpu_tensor_binding_سائز(b, alignment) &&
           b_انحراف < a_انحراف + ggml_webgpu_tensor_binding_سائز(a, alignment);
}

ساکن منطقی ggml_webgpu_tensor_binding_overlap_range(ثابت webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_بفر_t         بفر,
                                                     size_t                        انحراف,
                                                     size_t                        size) {
    اگر (tensor->بفر != بفر) {
        واپس جھوٹ;
    }

    ثابت size_t alignment     = global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment;
    ثابت size_t tensor_انحراف = ggml_webgpu_tensor_align_انحراف(tensor, alignment);
    واپس tensor_انحراف < انحراف + size && انحراف < tensor_انحراف + ggml_webgpu_tensor_binding_سائز(tensor, alignment);
}

ساخت ggml_webgpu_merged_binding_range {
    size_t انحراف;
    size_t size;
};

ساکن ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_انحراف = SIZE_MAX;
    size_t merged_انتہا    = 0;

    برائے (ggml_tensor * tensor : tensors) {
        ثابت size_t bind_انحراف = ggml_webgpu_tensor_align_انحراف(ctx, tensor);
        ثابت size_t bind_انتہا    = bind_انحراف + ggml_webgpu_tensor_binding_سائز(ctx, tensor);

        merged_انحراف = std::min(merged_انحراف, bind_انحراف);
        merged_انتہا    = std::max(merged_انتہا, bind_انتہا);
    }

    واپس { merged_انحراف, merged_انتہا - merged_انحراف };
}

ساکن uint32_t ggml_webgpu_tensor_merged_element_انحراف(ثابت ggml_tensor *                      tensor,
                                                         ثابت ggml_webgpu_merged_binding_range & merged_range) {
    واپس (uint32_t) ((ggml_webgpu_tensor_انحراف(tensor) - merged_range.انحراف) / ggml_type_سائز(tensor->type));
}

ساکن wgpu::Bind_گروہ_اندراج ggml_webgpu_make_bind_گروہ_اندراج(uint32_t     binding,
                                                              wgpu::بفر_2 بفر,
                                                              uint64_t     انحراف,
                                                              uint64_t     size) {
    wgpu::Bind_گروہ_اندراج اندراج_2 = {};
    اندراج_2.binding              = binding;
    اندراج_2.بفر               = std::move(بفر);
    اندراج_2.انحراف               = انحراف;
    اندراج_2.size                 = size;
    واپس اندراج_2;
}

ساکن wgpu::Bind_گروہ_اندراج ggml_webgpu_make_tensor_bind_گروہ_اندراج(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    واپس ggml_webgpu_make_bind_گروہ_اندراج(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_انحراف(ctx, tensor),
                                             ggml_webgpu_tensor_binding_سائز(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

سانچہ <قسم_نام T>
ساکن خالی ggml_backend_webgpu_check_wait_حالت(wgpu::Wait_حالت wait_حالت,
                                                  T                callback_حالت,
                                                  T                success_حالت,
                                                  ثابت حرف *     wait_name,
                                                  ثابت حرف *     failure_name,
                                                  ثابت حرف *     callback_پیغام) {
    اگر (wait_حالت == wgpu::Wait_حالت::Timed_برآمد) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    اگر (wait_حالت == wgpu::Wait_حالت::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    اگر (callback_حالت != success_حالت) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<صحیح>(callback_حالت),
                   callback_پیغام);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
ساکن uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    واپس UINT32_MAX;
}

ساکن uint32_t ggml_backend_webgpu_get_command_submit_batch_سائز() {
    واپس WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_سائز;
}

ساکن خالی ggml_backend_webgpu_wait_قطار(webgpu_global_context & ctx) {
    wgpu::قطار_Work_مکمل_حالت callback_حالت = wgpu::قطار_Work_مکمل_حالت::Error;
    std::string               callback_پیغام;

    ثابت wgpu::Wait_حالت wait_حالت = ctx->instance.WaitAny(
        ctx->queue.On_Submitted_Work_مکمل(
            wgpu::Callback_منوال::AllowSpontaneous,
            [&callback_حالت, &callback_پیغام](wgpu::قطار_Work_مکمل_حالت حالت_2, wgpu::StringView پیغام) {
                callback_حالت  = حالت_2;
                callback_پیغام = std::string(پیغام);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_حالت(wait_حالت, callback_حالت, wgpu::قطار_Work_مکمل_حالت::Success,
                                          "Queue wait", "Queue work", callback_پیغام.c_str());
}

ساکن خالی ggml_backend_webgpu_map_بفر(webgpu_global_context & ctx,
                                           wgpu::بفر_2 &          بفر,
                                           wgpu::Map_منوال           mode,
                                           size_t                  انحراف,
                                           size_t                  size) {
    wgpu::Map_Async_حالت callback_حالت = wgpu::Map_Async_حالت::Error;
    std::string          callback_پیغام;

    ثابت wgpu::Wait_حالت wait_حالت = ctx->instance.WaitAny(
        بفر.MapAsync(mode, انحراف, size, wgpu::Callback_منوال::AllowSpontaneous,
                        [&callback_حالت, &callback_پیغام](wgpu::Map_Async_حالت حالت_2, wgpu::StringView پیغام) {
                            callback_حالت  = حالت_2;
                            callback_پیغام = std::string(پیغام);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_حالت(wait_حالت, callback_حالت, wgpu::Map_Async_حالت::Success,
                                          "Buffer map wait", "Buffer map", callback_پیغام.c_str());
}

ساکن خالی ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ثابت wgpu::Command_بفر commands,
                                                uint32_t &                عدد_inflight_batches) {
    اگر (عدد_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_قطار(ctx->global_ctx);
        عدد_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    عدد_inflight_batches++;
}

#اگر_تعریف_شدہ GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
ساکن خالی ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_بفر_مقصد_بفر(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_سائز());
    wgpu::Command_بفر commands = encoder.اختتام_2();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_بفر(ctx, ctx->debug_host_buf, wgpu::Map_منوال::Read, 0, ctx->debug_host_buf.Get_سائز());
    ثابت اعشاری * debug_ڈیٹا = (ثابت اعشاری *) ctx->debug_host_buf.GetConstMappedRange();
    std::خارج << "debug[0]: " << debug_ڈیٹا[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#اگر_ختم

ساکن webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ثابت std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            نتیجہ = {};
    std::vector<wgpu::Bind_گروہ> bind_groups;
    std::vector<size_t>          param_offsets;
    نتیجہ.عدد_kernels = dispatches.size();

    برائے (size_t i = 0; i < dispatches.size(); i++) {
        ثابت webgpu_dispatch_desc & dispatch     = dispatches[i];
        ثابت size_t                 param_سائز   = dispatch.params.size() * حجم(uint32_t);
        ثابت size_t                 param_انحراف = ctx->param_arena.alloc_خانہ(param_سائز);

        std::vector<wgpu::Bind_گروہ_اندراج> entries            = dispatch.bind_گروہ_entries;
        uint32_t                          params_binding_عدد = entries.size();
        entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(params_binding_عدد, ctx->param_arena.بفر, param_انحراف,
                                                            ctx->param_arena.خانہ_سائز));

        wgpu::Bind_گروہ_Descriptor bind_گروہ_desc;
        bind_گروہ_desc.layout     = dispatch.pipeline.pipeline.Get_Bind_گروہ_Layout(0);
        bind_گروہ_desc.اندراج_گنتی = entries.size();
        bind_گروہ_desc.entries    = entries.ڈیٹا();
        bind_گروہ_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.Create_Bind_گروہ(&bind_گروہ_desc));
        param_offsets.push_back(param_انحراف);
    }

    برائے (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_بفر(ctx->param_arena.بفر, param_offsets[i], dispatches[i].params.ڈیٹا(),
                                           dispatches[i].params.size() * حجم(uint32_t));
    }

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    برائے (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_گنتی + 2 <= WEBGPU_MAX_PROFILE_QUERY_گنتی);
        ثابت uint32_t query_begin = ctx->profile_timestamp_query_گنتی++;
        ثابت uint32_t query_انتہا   = ctx->profile_timestamp_query_گنتی++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.query_سیٹ                    = ctx->profile_timestamp_query_سیٹ;
        ts_writes.beginning_کا_Pass_Write_اشاریہ   = query_begin;
        ts_writes.انتہا_کا_Pass_Write_اشاریہ         = query_انتہا;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->فعال_command_encoder.BeginComputePass(&pass_desc);

        pass.سیٹ_Pipeline(dispatches[i].pipeline.pipeline);
        pass.سیٹ_Bind_گروہ(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.پہلا, dispatches[i].workgroups.second, 1);
        pass.انتہا_2();
        نتیجہ.pipeline_نام.push_back(dispatches[i].pipeline.name);
    }
#ورنہ
    برائے (size_t i = 0; i < dispatches.size(); i++) {
        اگر (ctx->batch_compute_passes) {
            ctx->فعال_compute_pass.سیٹ_Pipeline(dispatches[i].pipeline.pipeline);
            ctx->فعال_compute_pass.سیٹ_Bind_گروہ(0, bind_groups[i]);
            ctx->فعال_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.پہلا, dispatches[i].workgroups.second,
                                                        1);
        } ورنہ {
            wgpu::ComputePassEncoder pass = ctx->فعال_command_encoder.BeginComputePass();
            pass.سیٹ_Pipeline(dispatches[i].pipeline.pipeline);
            pass.سیٹ_Bind_گروہ(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.پہلا, dispatches[i].workgroups.second, 1);
            pass.انتہا_2();
        }
    }
#اگر_ختم

    واپس نتیجہ;
}

ساکن webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::Bind_گروہ_اندراج> bind_گروہ_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    واپس ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_گروہ_entries), { wg_x, wg_y } },
    });
}

ساکن خالی ggml_backend_webgpu_بفر_memset(webgpu_global_context & ctx,
                                              wgpu::بفر_2 &          buf,
                                              uint32_t                value,
                                              size_t                  انحراف,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) انحراف, (uint32_t) size, value };
    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_bind_گروہ_اندراج(0, buf, 0, buf.Get_سائز()) };
    size_t                            بائٹیں_فی_wg =
        ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup * ctx->capabilities.memset_بائٹیں_فی_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, بائٹیں_فی_wg);

    ctx->queue.Write_بفر(ctx->memset_params_buf, 0, params.ڈیٹا(), params.size() * حجم(uint32_t));

    wgpu::Bind_گروہ_اندراج params_اندراج = {};
    params_اندراج.binding              = 1;
    params_اندراج.بفر               = ctx->memset_params_buf;
    params_اندراج.انحراف               = 0;
    params_اندراج.size                 = WEBGPU_PARAMS_BUF_سائز_بائٹیں;
    entries.push_back(params_اندراج);

    wgpu::Bind_گروہ_Descriptor bind_گروہ_desc;
    bind_گروہ_desc.layout     = ctx->memset_pipeline.pipeline.Get_Bind_گروہ_Layout(0);
    bind_گروہ_desc.اندراج_گنتی = entries.size();
    bind_گروہ_desc.entries    = entries.ڈیٹا();
    bind_گروہ_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::Bind_گروہ bind_گروہ = ctx->device.Create_Bind_گروہ(&bind_گروہ_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.سیٹ_Pipeline(ctx->memset_pipeline.pipeline);
    pass.سیٹ_Bind_گروہ(0, bind_گروہ);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.انتہا_2();

    wgpu::Command_بفر              command  = encoder.اختتام_2();
    std::vector<wgpu::Command_بفر> commands = { command };
    ctx->queue.Submit(commands.size(), commands.ڈیٹا());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

ساکن ثابت حرف * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    واپس ctx->name.c_str();
}

ساکن خالی ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#اگر_تعریف_شدہ GGML_WEBGPU_CPU_PROFILE
    std::خارج << "\n[ggml_webgpu cpu profiling summary]\n";
    دوہرا کل_cpu = 0.0;
    برائے (ثابت خودکار & kv : ctx->webgpu_ctx->global_ctx->cpu_وقت_ms) {
        کل_cpu += kv.second;
    }
    std::خارج << "ggml_webgpu: total cpu time: " << کل_cpu << " ms\n";
    std::خارج << "ggml_webgpu: cpu breakdown:\n";
    برائے (ثابت خودکار & kv : ctx->webgpu_ctx->global_ctx->cpu_وقت_ms) {
        دوہرا pct = (کل_cpu > 0.0) ? (kv.second / کل_cpu * 100.0) : 0.0;
        std::خارج << "ggml_webgpu:  " << kv.پہلا << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    اگر (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::خارج << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    برائے (ثابت خودکار & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        دوہرا pct = (کل_cpu > 0.0) ? (kv.second / کل_cpu * 100.0) : 0.0;
        std::خارج << "ggml_webgpu:  " << kv.پہلا << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#اگر_ختم

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    std::خارج << "\n[ggml_webgpu gpu profiling summary]\n";
    دوہرا کل_gpu = 0.0;
    برائے (ثابت خودکار & kv : ctx->webgpu_ctx->shader_gpu_وقت_ms) {
        کل_gpu += kv.second;
    }
    std::خارج << "ggml_webgpu: total gpu time (all shaders): " << کل_gpu << " ms\n";
    std::خارج << "\nggml_webgpu: gpu breakdown:\n";
    برائے (ثابت خودکار & kv : ctx->webgpu_ctx->shader_gpu_وقت_ms) {
        دوہرا pct = (کل_gpu > 0.0) ? (kv.second / کل_gpu * 100.0) : 0.0;
        std::خارج << "ggml_webgpu:  " << kv.پہلا << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#اگر_ختم

#اگر defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::خارج << "ggml_webgpu: gpu/cpu ratio: " << (کل_cpu > 0.0 ? کل_gpu / کل_cpu : 0.0) << "\n";
#اگر_ختم

    حذف ctx;
    حذف backend;
}

ساکن webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_سائز(src->type)), (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)), (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_سائز(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ne, decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_سیٹ(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_سیٹ_pipeline(shader_lib_ctx);

    خودکار *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ثابت منطقی inplace   = decisions->inplace;

    ثابت uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    ثابت uint32_t dst_type_سائز = (uint32_t) ggml_type_سائز(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (((ثابت int32_t *) dst->op_params)[3] / dst_type_سائز),

        (uint32_t) (src1->nb[0] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),

        1u,
        (uint32_t) (((ثابت int32_t *) dst->op_params)[0] / dst_type_سائز),
        (uint32_t) (((ثابت int32_t *) dst->op_params)[1] / dst_type_سائز),
        (uint32_t) (((ثابت int32_t *) dst->op_params)[2] / dst_type_سائز),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries;
    uint32_t                          binding_اشاریہ = 0;
    اگر (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0));
        binding_اشاریہ++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_اشاریہ, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_اشاریہ + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ne, decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)),
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

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ne, decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.wg_mem_حد_بندی_بائٹیں = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_سائز;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst),
    };

    ثابت uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_سائز);
    ثابت uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    ثابت int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ثابت int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ثابت int32_t d1 = ggml_get_op_params_i32(dst, 5);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),

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

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
ساکن webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
    ثابت int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t p0 = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t p1 = ggml_get_op_params_i32(dst, 3);
    ثابت int32_t d0 = ggml_get_op_params_i32(dst, 4);
    ثابت int32_t d1 = ggml_get_op_params_i32(dst, 5);

    // Scalar params matching conv2d_dw.wgsl (weight src0 [KW,KH,1,C], input src1, output dst).
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),

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

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ثابت منطقی      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    ثابت int32_t s0    = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t s1    = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t p0    = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t p1    = ggml_get_op_params_i32(dst, 3);
    ثابت int32_t d0    = ggml_get_op_params_i32(dst, 4);
    ثابت int32_t d1    = ggml_get_op_params_i32(dst, 5);
    ثابت منطقی    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    ثابت uint32_t KW = src0->ne[0];
    ثابت uint32_t KH = is_2D ? src0->ne[1] : 1;
    ثابت uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    ثابت uint32_t IW = src1->ne[0];
    ثابت uint32_t IH = is_2D ? src1->ne[1] : 1;
    ثابت uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    ثابت uint32_t OW = dst->ne[1];
    ثابت uint32_t OH = is_2D ? dst->ne[2] : 1;

    ثابت uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_سائز(src1->type));
    ثابت uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)) : 0;
    ثابت uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type));
    ثابت uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type));

    ثابت uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_سائز(dst->type));
    ثابت uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type));
    ثابت uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)) : 0;
    ثابت uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),

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

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->علامات_فی_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst),
    };

    ثابت uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->بلاک_سائز_3);
    ثابت uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_ssm_جائزہ(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_سائز        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    اگر (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    اگر (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    اگر (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_جائزہ_pipeline(shader_lib_ctx, xbc_overlap);
    خودکار *          decisions = static_cast<ggml_webgpu_ssm_جائزہ_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t انحراف_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type));
    uint32_t انحراف_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_سائز(src2->type));
    uint32_t انحراف_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_سائز(src3->type));
    uint32_t انحراف_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_سائز(src4->type));
    uint32_t انحراف_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_سائز(src5->type));
    uint32_t انحراف_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_سائز(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    اگر (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        انحراف_x         = ggml_webgpu_tensor_merged_element_انحراف(src1, xbc_merged_range);
        انحراف_B         = ggml_webgpu_tensor_merged_element_انحراف(src4, xbc_merged_range);
    } ورنہ اگر (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        انحراف_B         = ggml_webgpu_tensor_merged_element_انحراف(src4, xbc_merged_range);
        انحراف_C         = ggml_webgpu_tensor_merged_element_انحراف(src5, xbc_merged_range);
    } ورنہ اگر (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        انحراف_x         = ggml_webgpu_tensor_merged_element_انحراف(src1, xbc_merged_range);
        انحراف_B         = ggml_webgpu_tensor_merged_element_انحراف(src4, xbc_merged_range);
        انحراف_C         = ggml_webgpu_tensor_merged_element_انحراف(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_انحراف = 0;
    size_t xbc_bind_سائز   = 0;
    اگر (xbc_overlap > 0) {
        xbc_bind_انحراف = xbc_merged_range.انحراف;
        xbc_bind_سائز   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        انحراف_x,
        انحراف_dt,
        انحراف_A,
        انحراف_B,
        انحراف_C,
        انحراف_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_سائز(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_سائز(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_سائز(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_سائز(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_سائز(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_سائز(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_سائز(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_سائز(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_سائز(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_عدد = 0;

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, src0),
    };
    // xbc_merged binding
    اگر (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(binding_عدد++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_انحراف, xbc_bind_سائز));
    }
    // x
    اگر (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, src1));
    }
    // B
    اگر (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, src4));
    }
    // C
    اگر (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد++, dst));

    ثابت uint32_t کل_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(کل_wg, max_wg_فی_dim, wg_x, wg_y);

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ثابت uint32_t s_v      = (uint32_t) src2->ne[0];
    ثابت uint32_t h        = (uint32_t) src2->ne[1];
    ثابت uint32_t n_علامات = (uint32_t) src2->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ثابت uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ثابت اعشاری    scale    = 1.0f / sqrtf((اعشاری) s_v);
    uint32_t       scale_u32;
    حافظہ_نقل(&scale_u32, &scale, حجم(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_علامات,
        n_seqs,
        s_v * h * n_علامات * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_سائز(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_سائز(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_سائز(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_سائز(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_سائز(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_سائز(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_سائز(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_سائز(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0), ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, src2), ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 4, src4), ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 6, dst),
    };

    اگر (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 7, dst_fuse));
    }

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

ساکن std::optional<webgpu_encoded_op> ggml_webgpu_سیٹ_صفیں(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    اگر (ggml_is_خالی_خانے(src) || ggml_is_خالی_خانے(idx)) {
        واپس std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_سیٹ_صفیں_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_سیٹ_صفیں_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_سائز(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)), (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)), (uint32_t) (idx->nb[0] / ggml_type_سائز(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_سائز(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_سائز(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst));

    اگر (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(3, ctx->سیٹ_صفیں_dev_error_buf, 0,
                                                            ctx->سیٹ_صفیں_dev_error_buf.Get_سائز()));
    }

    uint32_t دھاگے;
    اگر (ggml_is_quantized(dst->type)) {
        ثابت uint32_t بلاکیں_فی_صف = src->ne[0] / ggml_blck_سائز(dst->type);
        دھاگے =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_بلاکیں ? (بلاکیں_فی_صف / 2) : بلاکیں_فی_صف);
    } ورنہ اگر (decisions->vec4) {
        دھاگے = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } ورنہ {
        دھاگے = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(دھاگے, decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
ساکن std::vector<wgpu::Constant_اندراج> ggml_webgpu_wg_سائز_اندراج(uint32_t wg_سائز) {
    std::vector<wgpu::Constant_اندراج> constants(1);
    constants[0].کلید   = "wg_size";
    constants[0].value = wg_سائز;
    واپس constants;
}

ساکن webgpu_encoded_op ggml_webgpu_get_صفیں(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ثابت منطقی float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = خالی_اشاریہ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت uint32_t انحراف_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type));
    ثابت uint32_t انحراف_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type));
    ثابت uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_سائز(src->type));
    ثابت uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_سائز(src->type));
    ثابت uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_سائز(src->type));

    ثابت منطقی vec4_aligned = انحراف_src % 4 == 0 && انحراف_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_صفیں_pipeline(shader_lib_ctx, vec4_aligned);
    خودکار *          decisions = static_cast<ggml_webgpu_get_صفیں_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { انحراف_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_سائز(idx->type)),
                                     انحراف_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_سائز(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_سائز(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_سائز(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst) };

    uint32_t بلاکیں_فی_صف = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t کل_صفیں     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t کل_دھاگے  = float_parallel ? بلاکیں_فی_صف * کل_صفیں : کل_صفیں;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(کل_دھاگے, decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن خالی ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_سائز        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ثابت size_t dst_انحراف           = ggml_webgpu_tensor_انحراف(dst);
    ثابت size_t q8_src1_align_انحراف = ROUNDUP_POW2(
        dst_انحراف + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment);
    ثابت size_t q8_src1_binding_سائز = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_گروہ_اندراج> q8_entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src1),
        ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_انحراف, q8_src1_binding_سائز)
    };

    خودکار q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_سائز     = q8_decisions->wg_سائز;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ثابت uint32_t wg_فی_vec     = (src0->ne[0] / 4 + (q8_wg_سائز - 1)) / q8_wg_سائز;
    ثابت uint32_t q8_کل_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_فی_vec;
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    compute_2d_workgroups(q8_کل_wg, max_wg_فی_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

ساکن webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    منطقی use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    منطقی use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_حاصل_ضرب,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_سائز              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_میٹرکس = ctx->global_ctx->capabilities.supports_subgroup_میٹرکس;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_سائز        = ctx->global_ctx->capabilities.min_subgroup_سائز;
    shader_lib_ctx.max_subgroup_سائز        = ctx->global_ctx->capabilities.max_subgroup_سائز;
    shader_lib_ctx.supports_dot_حاصل_ضرب     = ctx->global_ctx->capabilities.supports_dot_حاصل_ضرب;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    ثابت منطقی src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    اگر (use_mat_vec) {
        اگر (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } ورنہ {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t انحراف_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type));
    uint32_t انحراف_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type));
    size_t   merged_انحراف = 0;
    size_t   merged_سائز   = 0;
    اگر (src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_انحراف = merged_range.انحراف;
        merged_سائز   = merged_range.size;
        انحراف_src0   = ggml_webgpu_tensor_merged_element_انحراف(src0, merged_range);
        انحراف_src1   = ggml_webgpu_tensor_merged_element_انحراف(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { انحراف_src0,
                                     انحراف_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::Bind_گروہ_اندراج> entries = {};
    اگر (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0));
        خودکار & mmvq_qq8_اندراج = dispatches[0].bind_گروہ_entries[1];
        entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_اندراج.انحراف,
                                                            mmvq_qq8_اندراج.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst));
    } ورنہ اگر (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src0), merged_انحراف, merged_سائز));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst));
    } ورنہ {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;

    اگر (use_mat_vec) {
        خودکار * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_فی_wg);
        uint32_t کل_wg      = output_groups * batches;
        compute_2d_workgroups(کل_wg, max_wg_فی_dim, wg_x, wg_y);
    } ورنہ {
        خودکار * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        اگر (decisions->use_subgroup_میٹرکس) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_میٹرکس_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_میٹرکس_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } ورنہ {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_سائز_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_سائز_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t کل_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(کل_wg, max_wg_فی_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    واپس ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساکن webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    ثابت uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ثابت uint32_t param_n_expert_مستعمل = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_سائز(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_مستعمل,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_انحراف(ctx, src0),
                                          ggml_webgpu_tensor_binding_سائز(ctx, src0)),
        ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_انحراف(ctx, src1),
                                          ggml_webgpu_tensor_binding_سائز(ctx, src1)),
        ggml_webgpu_make_bind_گروہ_اندراج(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_انحراف(ctx, src2),
                                          ggml_webgpu_tensor_binding_سائز(ctx, src2)),
        ggml_webgpu_make_bind_گروہ_اندراج(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_انحراف(ctx, dst),
                                          ggml_webgpu_tensor_binding_سائز(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    خودکار * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_فی_wg);
    uint32_t       کل_wg       = output_groups * param_n_expert_مستعمل;
    compute_2d_workgroups(کل_wg, max_wg_فی_dim, wg_x, wg_y);

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
ساکن uint32_t ggml_webgpu_mul_mat_id_max_tiles(ثابت ggml_tensor * dst) {
    ثابت uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    ثابت uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    واپس CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_سائز_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
ساکن size_t ggml_webgpu_mul_mat_id_tiles_nbytes(ثابت ggml_tensor * dst) {
    واپس (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * حجم(uint32_t);
}

ساکن webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    اگر (dst->ne[2] == 1) {
        واپس ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ثابت uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ثابت uint32_t param_n_expert_مستعمل = (uint32_t) dst->ne[1];
    ثابت uint32_t param_n_علامات      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_سائز(src2->type)),
        param_n_expert,
        param_n_expert_مستعمل,
        param_n_علامات,
        (uint32_t) (src2->nb[1] / ggml_type_سائز(src2->type)),
    };

    ثابت size_t dst_انحراف       = ggml_webgpu_tensor_انحراف(dst);
    ثابت size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * حجم(uint32_t);
    ثابت size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ثابت size_t ids_cache_align_انحراف = ROUNDUP_POW2(
        dst_انحراف + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment);
    ثابت size_t ids_tiles_align_انحراف =
        ROUNDUP_POW2(ids_cache_align_انحراف + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment);

    ثابت size_t ids_cache_bind_سائز = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ثابت size_t ids_tiles_bind_سائز = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::Bind_گروہ_اندراج> gather_entries = {
        ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_انحراف(ctx, src2),
                                          ggml_webgpu_tensor_binding_سائز(ctx, src2)),
        ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_انحراف, ids_cache_bind_سائز),
        ggml_webgpu_make_bind_گروہ_اندراج(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_انحراف, ids_tiles_bind_سائز),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_مستعمل,
        param_n_علامات,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::Bind_گروہ_اندراج> main_entries = {
        ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_انحراف(ctx, src0),
                                          ggml_webgpu_tensor_binding_سائز(ctx, src0)),
        ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_انحراف(ctx, src1),
                                          ggml_webgpu_tensor_binding_سائز(ctx, src1)),
        ggml_webgpu_make_bind_گروہ_اندراج(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_انحراف(ctx, dst),
                                          ggml_webgpu_tensor_binding_سائز(ctx, dst)),
        ggml_webgpu_make_bind_گروہ_اندراج(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_انحراف, ids_cache_bind_سائز),
        ggml_webgpu_make_bind_گروہ_اندراج(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_انحراف, ids_tiles_bind_سائز),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    خودکار * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_سائز_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_سائز_N);

    ثابت uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_سائز_m;
    ثابت uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ثابت uint32_t کل_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    واپس ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساخت ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_گروہ_اندراج> entries;
    size_t                            kv_bind_انحراف = 0;
    size_t                            kv_bind_سائز   = 0;
    منطقی                              has_نقاب       = جھوٹ;
    منطقی                              has_sinks      = جھوٹ;
    منطقی                              kv_overlap     = جھوٹ;
};

ساکن منطقی ggml_webgpu_flash_attn_use_vec_راستہ(ثابت webgpu_global_context & global_ctx,
                                                ثابت ggml_tensor *           Q,
                                                ثابت ggml_tensor *           K,
                                                ثابت ggml_tensor *           V) {
    ثابت size_t storage_انحراف_alignment = global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment;

    ثابت منطقی k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_انحراف_alignment);
    ثابت منطقی v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_انحراف_alignment);

    ثابت uint32_t k_vec_سر_align =
        ggml_is_quantized(K->type) ? ggml_blck_سائز(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی;
    ثابت uint32_t v_vec_سر_align =
        ggml_is_quantized(V->type) ? ggml_blck_سائز(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی;
    ثابت منطقی kv_vec_سر_dims_aligned = Q->ne[0] % k_vec_سر_align == 0 && V->ne[0] % v_vec_سر_align == 0;

    واپس global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_سر_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

ساکن ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    نقاب,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    اعشاری scale         = ggml_get_op_params_f32(dst, 0);
    اعشاری max_bias      = ggml_get_op_params_f32(dst, 1);
    اعشاری logit_softcap = ggml_get_op_params_f32(dst, 2);
    اگر (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    اعشاری n_سر_log2 = اعشاری(1u << (uint32_t) floor(log2(Q->ne[2])));
    اعشاری m0          = powf(2.0f, -(max_bias) / n_سر_log2);
    اعشاری m1          = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = نقاب;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_میٹرکس = ctx->global_ctx->capabilities.supports_subgroup_میٹرکس;
    op.shader_lib_ctx.max_wg_سائز              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    op.shader_lib_ctx.wg_mem_حد_بندی_بائٹیں       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_سائز;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_سائز        = ctx->global_ctx->capabilities.min_subgroup_سائز;
    op.shader_lib_ctx.max_subgroup_سائز        = ctx->global_ctx->capabilities.max_subgroup_سائز;

    op.has_نقاب   = نقاب != خالی_اشاریہ;
    op.has_sinks  = sinks != خالی_اشاریہ;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t انحراف_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_سائز(K->type));
    uint32_t انحراف_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_سائز(V->type));
    اگر (op.kv_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_انحراف                                   = merged_range.انحراف;
        op.kv_bind_سائز                                     = merged_range.size;
        انحراف_k                                            = ggml_webgpu_tensor_merged_element_انحراف(K, merged_range);
        انحراف_v                                            = ggml_webgpu_tensor_merged_element_انحراف(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_سائز(Q->type)),
        انحراف_k,
        انحراف_v,
        op.has_نقاب ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, نقاب) / ggml_type_سائز(نقاب->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_سائز(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_سائز(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_سائز(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_سائز(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_سائز(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_سائز(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_سائز(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_سائز(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_سائز(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_سائز(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_نقاب ? (uint32_t) (نقاب->nb[3] / ggml_type_سائز(نقاب->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_سر_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, Q),
    };
    اگر (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(K), op.kv_bind_انحراف, op.kv_bind_سائز));
    } ورنہ {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, V));
    }
    uint32_t binding_اشاریہ = op.kv_overlap ? 2u : 3u;
    اگر (op.has_نقاب) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_اشاریہ++, نقاب));
    }
    اگر (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_اشاریہ++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_اشاریہ++, dst));

    واپس op;
}

ساکن uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ثابت uint64_t kv_وسعت = (uint64_t) kv_tile;
    جب ((2u * nwg * kv_وسعت) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    واپس std::min(nwg, vec_nwg_cap);
}

ساکن webgpu_encoded_op ggml_webgpu_flash_attn_براہ_راست(webgpu_context & ctx, ثابت ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_فی_سر = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_فی_سر * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    واپس ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

ساکن webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             نقاب,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    خودکار *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::بفر_2 blk_buf         = {};
    uint64_t     blk_سائز_بائٹیں  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_گنتی = 0;

    ثابت uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_سائز;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ثابت uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ثابت منطقی     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_بنیاد  = 0;
    uint64_t     tmp_سائز_بائٹیں  = 0;
    wgpu::بفر_2 tmp_buf         = {};
    uint64_t     tmp_bind_انحراف = 0;
    uint64_t     tmp_bind_سائز   = 0;
    ثابت size_t align_بائٹیں     = ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment;
    ثابت size_t dst_انحراف      = ggml_webgpu_tensor_انحراف(dst);
    size_t       scratch_انحراف  = ROUNDUP_POW2(dst_انحراف + ggml_nbytes(dst), align_بائٹیں);

    اگر (use_vec_reduce) {
        ثابت uint64_t tmp_ڈیٹا_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ثابت uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_بنیاد                 = tmp_ڈیٹا_elems;
        tmp_سائز_بائٹیں =
            ROUNDUP_POW2((tmp_ڈیٹا_elems + tmp_stats_elems) * حجم(اعشاری), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_بنیاد <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_انحراف = scratch_انحراف;
        tmp_bind_سائز   = tmp_سائز_بائٹیں;
        scratch_انحراف  = ROUNDUP_POW2(scratch_انحراف + tmp_سائز_بائٹیں, align_بائٹیں);
    } ورنہ {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_سائز_بائٹیں  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_انحراف = scratch_انحراف;
        tmp_bind_سائز   = tmp_سائز_بائٹیں;
        scratch_انحراف  = ROUNDUP_POW2(scratch_انحراف + tmp_سائز_بائٹیں, align_بائٹیں);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::Bind_گروہ_اندراج> blk_entries;
    اگر (op.has_نقاب) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ثابت uint32_t stride_mask3 = (uint32_t) (نقاب->nb[3] / ggml_type_سائز(نقاب->type));
        blk_batch_گنتی             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ثابت uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_گنتی;
        blk_سائز_بائٹیں              = ROUNDUP_POW2(blk_elems * حجم(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        ثابت ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, نقاب) / ggml_type_سائز(نقاب->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(نقاب),
                                              ggml_webgpu_tensor_align_انحراف(ctx, نقاب),
                                              ggml_webgpu_tensor_binding_سائز(ctx, نقاب)),
            ggml_webgpu_make_bind_گروہ_اندراج(1, blk_buf, scratch_انحراف, blk_سائز_بائٹیں),
        };
        scratch_انحراف = ROUNDUP_POW2(scratch_انحراف + blk_سائز_بائٹیں, align_بائٹیں);
    }

    std::vector<uint32_t> بانٹو_params = op.params;
    اگر (op.has_نقاب) {
        بانٹو_params.push_back(0u);                     // blk_base
        بانٹو_params.push_back(blk_nblk0);              // blk_nblk0
        بانٹو_params.push_back(blk_nblk1);              // blk_nblk1
    }
    بانٹو_params.push_back(0u);                         // tmp_data_base
    بانٹو_params.push_back((uint32_t) tmp_stats_بنیاد);  // tmp_stats_base
    بانٹو_params.push_back(nwg);                        // nwg

    std::vector<wgpu::Bind_گروہ_اندراج> بانٹو_entries = {
        ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_انحراف(ctx, Q),
                                          ggml_webgpu_tensor_binding_سائز(ctx, Q)),
    };
    اگر (op.kv_overlap) {
        بانٹو_entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(K), op.kv_bind_انحراف, op.kv_bind_سائز));
    } ورنہ {
        بانٹو_entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_انحراف(ctx, K),
                                                                  ggml_webgpu_tensor_binding_سائز(ctx, K)));
        بانٹو_entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_انحراف(ctx, V),
                                                                  ggml_webgpu_tensor_binding_سائز(ctx, V)));
    }
    uint32_t بانٹو_binding_اشاریہ = op.kv_overlap ? 2u : 3u;
    اگر (op.has_نقاب) {
        بانٹو_entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(بانٹو_binding_اشاریہ++, ggml_webgpu_tensor_buf(نقاب),
                                                                  ggml_webgpu_tensor_align_انحراف(ctx, نقاب),
                                                                  ggml_webgpu_tensor_binding_سائز(ctx, نقاب)));
    }
    اگر (op.has_sinks) {
        بانٹو_entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(بانٹو_binding_اشاریہ++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_انحراف(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_سائز(ctx, sinks)));
    }
    اگر (op.has_نقاب) {
        بانٹو_entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(بانٹو_binding_اشاریہ++, blk_buf, blk_entries[1].انحراف, blk_سائز_بائٹیں));
    }
    بانٹو_entries.push_back(
        ggml_webgpu_make_bind_گروہ_اندراج(بانٹو_binding_اشاریہ++, tmp_buf, tmp_bind_انحراف, tmp_bind_سائز));
    بانٹو_entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(بانٹو_binding_اشاریہ++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_انحراف(ctx, dst),
                                                              ggml_webgpu_tensor_binding_سائز(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::Bind_گروہ_اندراج> reduce_entries;
    اگر (use_vec_reduce) {
        ثابت uint32_t reduce_sg_سائز = ctx->global_ctx->capabilities.max_subgroup_سائز;
        ثابت uint32_t reduce_wg_سائز = std::max(
            reduce_sg_سائز,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_سائز,
                                          ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_سائز                    = reduce_wg_سائز;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_بنیاد,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_گروہ_اندراج(0, tmp_buf, tmp_bind_انحراف, tmp_سائز_بائٹیں),
            ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_انحراف(ctx, dst),
                                              ggml_webgpu_tensor_binding_سائز(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ثابت uint64_t بانٹو_wg_کل = (uint64_t) wg_x * nwg;
    GGML_ASSERT(بانٹو_wg_کل <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    اگر (op.has_نقاب) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_گنتی }
        });
    }
    dispatches.push_back({
        pipeline, std::move(بانٹو_params), std::move(بانٹو_entries), { (uint32_t) بانٹو_wg_کل, 1u }
    });
    اگر (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    واپس ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساکن webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    نقاب,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, نقاب, sinks, dst);
    اگر (ggml_webgpu_flash_attn_use_vec_راستہ(ctx->global_ctx, Q, K, V)) {
        واپس ggml_webgpu_flash_attn_vec(ctx, Q, K, V, نقاب, sinks, dst, std::move(op));
    }
    واپس ggml_webgpu_flash_attn_براہ_راست(ctx, op);
}

ساکن webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقی is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = خالی_اشاریہ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    خودکار *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ثابت منطقی inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_سائز(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    اگر (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        انتخاب (unary_op) {
            حالت GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    اعشاری alpha_n = ggml_get_op_params_f32(dst, 1);
                    اعشاری alpha_p = ggml_get_op_params_f32(dst, 2);
                    اعشاری beta    = ggml_get_op_params_f32(dst, 3);
                    اعشاری eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    توڑو;
                }
            طےشدہ:
                توڑو;
        }
    } ورنہ اگر (dst->op == GGML_OP_CLAMP) {
        اعشاری clamp_min = ggml_get_op_params_f32(dst, 0);
        اعشاری clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } ورنہ اگر (dst->op == GGML_OP_FILL) {
        اعشاری fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, effective_src),
    };
    اگر (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_ثنائی_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقی      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_ثنائی_pipeline(shader_lib_ctx, src_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_ثنائی_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_انحراف = ggml_webgpu_tensor_align_انحراف(ctx, src0);
    size_t src1_webgpu_tensor_align_انحراف = ggml_webgpu_tensor_align_انحراف(ctx, src1);

    uint32_t انحراف_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type));
    uint32_t انحراف_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type));
    size_t   merged_انحراف = 0;
    size_t   merged_سائز   = 0;
    اگر (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_انحراف = merged_range.انحراف;
        merged_سائز   = merged_range.size;
        انحراف_src0   = ggml_webgpu_tensor_merged_element_انحراف(src0, merged_range);
        انحراف_src1   = ggml_webgpu_tensor_merged_element_انحراف(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        انحراف_src0,
        انحراف_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries;

    اگر (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src0), merged_انحراف, merged_سائز));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst));
    } ورنہ {
        entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_انحراف,
                                                            ggml_webgpu_tensor_binding_سائز(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_انحراف,
                                                            ggml_webgpu_tensor_binding_سائز(ctx, src1)));
        اگر (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_جوڑو_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_جوڑو_id_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_سائز(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_سائز(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_سائز(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, src2));

    اگر (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       کل_wg       = ggml_nrows(dst);
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    compute_2d_workgroups(کل_wg, max_wg_فی_dim, wg_x, wg_y);

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    اگر (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        واپس {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقی      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_ثنائی_shader_decisions *>(pipeline.context.get());

    uint32_t انحراف_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type));
    uint32_t انحراف_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type));
    size_t   merged_انحراف = 0;
    size_t   merged_سائز   = 0;
    اگر (decisions->src_overlap) {
        اگر (ggml_nbytes(src0) == 0) {
            merged_انحراف = ggml_webgpu_tensor_align_انحراف(ctx, src1);
            merged_سائز   = ggml_webgpu_tensor_binding_سائز(ctx, src1);
            انحراف_src0   = 0;
            انحراف_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type));
        } ورنہ اگر (ggml_nbytes(src1) == 0) {
            merged_انحراف = ggml_webgpu_tensor_align_انحراف(ctx, src0);
            merged_سائز   = ggml_webgpu_tensor_binding_سائز(ctx, src0);
            انحراف_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type));
            انحراف_src1   = 0;
        } ورنہ {
            ثابت ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_انحراف = merged_range.انحراف;
            merged_سائز   = merged_range.size;
            انحراف_src0   = ggml_webgpu_tensor_merged_element_انحراف(src0, merged_range);
            انحراف_src1   = ggml_webgpu_tensor_merged_element_انحراف(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     انحراف_src0,
                                     انحراف_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_سائز(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {};
    اگر (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src0), merged_انحراف, merged_سائز));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst));
    } ورنہ {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ne, decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_سائز(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ne, decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    اگر (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } ورنہ اگر (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } ورنہ {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t انحراف_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_سائز(rn_src->type));
    uint32_t انحراف_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_سائز(mul_src->type));
    size_t merged_انحراف = 0;
    size_t merged_سائز   = 0;

    std::vector<uint32_t> params = {
        انحراف_rn_src,
        انحراف_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_سائز(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_سائز(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_سائز(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_سائز(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_سائز(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_سائز(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
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

    std::vector<wgpu::Bind_گروہ_اندراج> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقی      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    اگر (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_انحراف  = merged_range.انحراف;
        merged_سائز    = merged_range.size;
        انحراف_rn_src  = ggml_webgpu_tensor_merged_element_انحراف(rn_src, merged_range);
        انحراف_mul_src = ggml_webgpu_tensor_merged_element_انحراف(mul_src, merged_range);
        params[0]      = انحراف_rn_src;
        params[1]      = انحراف_mul_src;
    }

    اگر (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, mul_src));
    } ورنہ اگر (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(rn_src), merged_انحراف, merged_سائز));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst));
    } ورنہ {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x,
                          wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_صف_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_صف_norm_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src) };
    اگر (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x,
                          wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت منطقی inplace         = decisions->inplace;
    ثابت صحیح  has_freq_ضربی_عدد = (src2 != خالی_اشاریہ);

    ثابت صحیح n_dims     = ((int32_t *) dst->op_params)[1];
    ثابت صحیح mode       = ((int32_t *) dst->op_params)[2];
    ثابت صحیح n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ثابت صحیح n_offs     = ((int32_t *) dst->op_params)[15];

    اعشاری freq_بنیاد;
    اعشاری freq_scale;
    اعشاری ext_ضربی_عدد;
    اعشاری attn_ضربی_عدد;
    اعشاری beta_fast;
    اعشاری beta_slow;
    حافظہ_نقل(&freq_بنیاد, (int32_t *) dst->op_params + 5, حجم(اعشاری));
    حافظہ_نقل(&freq_scale, (int32_t *) dst->op_params + 6, حجم(اعشاری));
    حافظہ_نقل(&ext_ضربی_عدد, (int32_t *) dst->op_params + 7, حجم(اعشاری));
    حافظہ_نقل(&attn_ضربی_عدد, (int32_t *) dst->op_params + 8, حجم(اعشاری));
    حافظہ_نقل(&beta_fast, (int32_t *) dst->op_params + 9, حجم(اعشاری));
    حافظہ_نقل(&beta_slow, (int32_t *) dst->op_params + 10, حجم(اعشاری));

    صحیح sections[4];
    حافظہ_نقل(sections, (int32_t *) dst->op_params + 11, 4 * حجم(صحیح));

    اعشاری theta_scale = powf(freq_بنیاد, -2.0f / n_dims);

    اعشاری corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنیاد, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)),
        src2 != خالی_اشاریہ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_سائز(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_ضربی_عدد),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_ضربی_عدد),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries     = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    اگر (has_freq_ضربی_عدد) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 2, src2));
    }
    اگر (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقی      src_overlap = src1 != خالی_اشاریہ && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    خودکار * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ثابت صحیح بانٹو = (src1 != خالی_اشاریہ);

    uint32_t انحراف_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type));
    uint32_t انحراف_src1 =
        src1 != خالی_اشاریہ ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)) : 0;
    size_t merged_انحراف = 0;
    size_t merged_سائز   = 0;
    اگر (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_انحراف = merged_range.انحراف;
        merged_سائز   = merged_range.size;
        انحراف_src0   = ggml_webgpu_tensor_merged_element_انحراف(src0, merged_range);
        انحراف_src1   = ggml_webgpu_tensor_merged_element_انحراف(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        انحراف_src0,
        انحراف_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
        src1 != خالی_اشاریہ ? (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        src1 != خالی_اشاریہ ? (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        src1 != خالی_اشاریہ ? (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  //حد_بندی
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries;
    uint32_t                          dst_binding = 1;
    اگر (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروہ_اندراج(0, ggml_webgpu_tensor_buf(src0), merged_انحراف, merged_سائز));
    } ورنہ {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src0));
    }
    اگر (بانٹو && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = خالی_اشاریہ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src) };

    اگر (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت منطقی inplace     = decisions->inplace;
    ثابت صحیح  has_نقاب    = (src1 != خالی_اشاریہ);
    ثابت صحیح  has_sink    = (src2 != خالی_اشاریہ);
    اعشاری      max_bias    = ggml_get_op_params_f32(dst, 1);
    اعشاری      n_سر_log2 = اعشاری(1u << (uint32_t) floor(log2(src0->ne[2])));
    اعشاری      m0          = powf(2.0f, -(max_bias) / n_سر_log2);
    اعشاری      m1          = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_سائز(src0->type)),
        has_نقاب ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_سائز(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_سائز(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_سائز(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_سائز(src0->type)),
        has_نقاب ? (uint32_t) (src1->nb[1] / ggml_type_سائز(src1->type)) : 0,
        has_نقاب ? (uint32_t) (src1->nb[2] / ggml_type_سائز(src1->type)) : 0,
        has_نقاب ? (uint32_t) (src1->nb[3] / ggml_type_سائز(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_نقاب ? (uint32_t) src1->ne[2] : 0,
        has_نقاب ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_سر_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::Bind_گروہ_اندراج> entries     = { ggml_webgpu_make_bind_گروہ_اندراج(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_انحراف(ctx, src0),
        ggml_webgpu_tensor_binding_سائز(ctx, src0)) };
    uint32_t                          binding_عدد = 1;
    اگر (has_نقاب) {
        entries.push_back(ggml_webgpu_make_bind_گروہ_اندراج(binding_عدد, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_انحراف(ctx, src1),
                                                            ggml_webgpu_tensor_binding_سائز(ctx, src1)));
        binding_عدد++;
    }
    اگر (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد, src2));
        binding_عدد++;
    }
    اگر (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, binding_عدد, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = ggml_nrows(dst);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

ساکن webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقی is_چوٹی_k = dst->op == GGML_OP_چوٹی_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = خالی_اشاریہ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.wg_mem_حد_بندی_بائٹیں = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_سائز;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    خودکار * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_ملاؤ_pipeline = ctx->shader_lib->get_argsort_ملاؤ_pipeline(shader_lib_ctx);

    ثابت uint32_t src_ne0 = (uint32_t) src->ne[0];
    ثابت uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ثابت uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_سائز);
    ثابت uint32_t بلاک_سائز_3 =
        is_چوٹی_k ? std::min(argsort_decisions->wg_سائز, (uint32_t) dst->ne[0]) : argsort_decisions->wg_سائز;
    uint32_t برآمد_ne0 = src_ne0;
    اگر (is_چوٹی_k) {
        اگر (npr > 1) {
            ثابت uint32_t آخری_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_سائز;
            برآمد_ne0                  = (npr - 1) * بلاک_سائز_3 + std::min(آخری_tile, بلاک_سائز_3);
        } ورنہ {
            برآمد_ne0 = بلاک_سائز_3;
        }
    }

    uint32_t ملاؤ_len    = بلاک_سائز_3;
    uint32_t ملاؤ_passes = 0;
    جب (ملاؤ_len < برآمد_ne0) {
        ملاؤ_len <<= 1;
        ملاؤ_passes++;
    }

    ثابت منطقی آغاز_اندر_tmp = (ملاؤ_passes % 2) == 1;

    ثابت size_t dst_انحراف = ggml_webgpu_tensor_انحراف(dst);
    ثابت size_t idx_nbytes = برآمد_ne0 * ggml_nrows(dst) * حجم(int32_t);
    ثابت size_t tmp_انحراف =
        ROUNDUP_POW2(dst_انحراف + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment);
    ثابت size_t tmp_binding_سائز = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ثابت size_t dst_binding_سائز =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ثابت uint32_t انحراف_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type));
    ثابت uint32_t انحراف_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type));
    ثابت uint32_t انحراف_tmp  = 0;
    ثابت uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_سائز(src->type));
    ثابت uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_سائز(src->type));
    ثابت uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_سائز(src->type));
    ثابت uint32_t stride_idx1 = برآمد_ne0;
    ثابت uint32_t stride_idx2 = برآمد_ne0 * (uint32_t) dst->ne[1];
    ثابت uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ثابت uint32_t init_انحراف       = آغاز_اندر_tmp ? انحراف_tmp : انحراف_dst;
    ثابت size_t   init_align_انحراف = آغاز_اندر_tmp ? tmp_انحراف : ggml_webgpu_tensor_align_انحراف(ctx, dst);
    ثابت size_t   init_binding_سائز = آغاز_اندر_tmp ? tmp_binding_سائز : dst_binding_سائز;

    std::vector<uint32_t> init_params = {
        انحراف_src,  init_انحراف, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], برآمد_ne0,
        بلاک_سائز_3,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ثابت uint32_t کل_wg_init  = npr * nrows;
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    compute_2d_workgroups(کل_wg_init, max_wg_فی_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::Bind_گروہ_اندراج> init_entries = {
        ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
        ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(dst), init_align_انحراف, init_binding_سائز)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    اگر (ملاؤ_passes == 0) {
        واپس ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    منطقی     اندر_is_tmp = آغاز_اندر_tmp;
    uint32_t len       = بلاک_سائز_3;
    جب (len < برآمد_ne0) {
        ثابت uint32_t nm = CEIL_DIV(برآمد_ne0, 2 * len);

        ثابت منطقی     برآمد_is_tmp  = !اندر_is_tmp;
        ثابت uint32_t انحراف_اندر   = اندر_is_tmp ? انحراف_tmp : انحراف_dst;
        ثابت uint32_t انحراف_برآمد  = برآمد_is_tmp ? انحراف_tmp : انحراف_dst;
        ثابت size_t   align_اندر    = اندر_is_tmp ? tmp_انحراف : ggml_webgpu_tensor_align_انحراف(ctx, dst);
        ثابت size_t   align_برآمد   = برآمد_is_tmp ? tmp_انحراف : ggml_webgpu_tensor_align_انحراف(ctx, dst);
        ثابت size_t   سائز_اندر     = اندر_is_tmp ? tmp_binding_سائز : dst_binding_سائز;
        ثابت size_t   سائز_برآمد    = برآمد_is_tmp ? tmp_binding_سائز : dst_binding_سائز;
        ثابت uint32_t چوٹی_k_برآمد   = (is_چوٹی_k && nm == 1) ? (uint32_t) dst->ne[0] : برآمد_ne0;
        ثابت uint32_t stride_out1 = چوٹی_k_برآمد;
        ثابت uint32_t stride_out2 = چوٹی_k_برآمد * (uint32_t) dst->ne[1];
        ثابت uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> ملاؤ_params = { انحراف_src,
                                               انحراف_اندر,
                                               انحراف_برآمد,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               برآمد_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               چوٹی_k_برآمد,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::Bind_گروہ_اندراج> ملاؤ_entries = {
            ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
            ggml_webgpu_make_bind_گروہ_اندراج(1, ggml_webgpu_tensor_buf(dst), align_اندر, سائز_اندر),
            ggml_webgpu_make_bind_گروہ_اندراج(2, ggml_webgpu_tensor_buf(dst), align_برآمد, سائز_برآمد)
        };

        uint32_t       wg_x_ملاؤ;
        uint32_t       wg_y_ملاؤ;
        ثابت uint32_t کل_wg_ملاؤ = nm * nrows;
        compute_2d_workgroups(کل_wg_ملاؤ, max_wg_فی_dim, wg_x_ملاؤ, wg_y_ملاؤ);

        dispatches.push_back({
            argsort_ملاؤ_pipeline, std::move(ملاؤ_params), std::move(ملاؤ_entries), { wg_x_ملاؤ, wg_y_ملاؤ }
        });

        len <<= 1;
        اندر_is_tmp = !اندر_is_tmp;
    }

    واپس ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساکن webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = خالی_اشاریہ;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        کل_wg = ggml_nrows(dst);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن webgpu_encoded_op ggml_webgpu_مجموع_صفیں(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقی                  کل_مجموع = dst->op == GGML_OP_مجموع;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),
                                     کل_مجموع ? 0 : (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)),
                                     کل_مجموع ? 0 : (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)),
                                     کل_مجموع ? 0 : (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)),
                                     کل_مجموع ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     کل_مجموع ? 1 : (uint32_t) src->ne[1],
                                     کل_مجموع ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_مجموع_صفیں_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = کل_مجموع ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساکن منطقی ggml_webgpu_can_fuse_rms_norm_mul(ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx) {
    اگر (!ggml_can_fuse(cgraph, گرہ_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        واپس جھوٹ;
    }

    // additional constraints specific to this fusion
    ثابت ggml_tensor * rms_norm = cgraph->nodes[گرہ_idx];
    ثابت ggml_tensor * mul      = cgraph->nodes[گرہ_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    اگر (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    اگر (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        واپس جھوٹ;
    }
    // rms_norm shader assumes contiguous rows
    اگر (!ggml_is_contiguous_صفیں(mul->src[0]) || !ggml_is_contiguous_صفیں(mul->src[1])) {
        واپس جھوٹ;
    }

    واپس سچ;
}

ساکن منطقی ggml_webgpu_can_fuse_gdn_cache(ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, صحیح & عدد_encoded_ops) {
    ثابت ggml_tensor * gdn = cgraph->nodes[گرہ_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    اگر (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        واپس جھوٹ;
    }

    ثابت ggml_tensor * src_v     = gdn->src[2];
    ثابت int64_t       S_v       = src_v->ne[0];
    ثابت int64_t       H         = src_v->ne[1];
    ثابت int64_t       n_علامات  = src_v->ne[2];
    ثابت int64_t       n_seqs    = src_v->ne[3];
    ثابت int64_t       D         = S_v * S_v * H;
    ثابت int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    ثابت int64_t       n_written = std::min<int64_t>(n_علامات, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ثابت size_t دم_off = ggml_صف_سائز(GGML_TYPE_F32, S_v * H * n_علامات * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ثابت ggml_tensor * cpy     = خالی_اشاریہ;
    صحیح                 cpy_idx = 0;
    برائے (صحیح j = گرہ_idx + 1; j < cgraph->n_nodes && cpy == خالی_اشاریہ; ++j) {
        ثابت ggml_tensor * n = cgraph->nodes[j];
        اگر (ggml_op_is_خالی_خانے(n->op) || ggml_is_خالی_خانے(n)) {
            جاری;
        }
        اگر (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            واپس جھوٹ;
        }
        cpy     = n;
        cpy_idx = j;
    }
    اگر (cpy == خالی_اشاریہ) {
        واپس جھوٹ;
    }

    ثابت ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ثابت ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    اگر (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != دم_off ||
        !ggml_is_contiguous(cpy_src)) {
        واپس جھوٹ;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ثابت std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    اگر (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->ڈیٹا == خالی_اشاریہ ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_سائز(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_صف_سائز(GGML_TYPE_F32, D)) {
        واپس جھوٹ;
    }

    عدد_encoded_ops = cpy_idx - گرہ_idx + 1;

    واپس سچ;
}

ساکن webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ثابت uint32_t        منوال_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_سائز(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_سائز(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_سائز(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_سائز(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_سائز(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_سائز(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_سائز(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_سائز(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_سائز(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_سائز(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     منوال_flags };

    std::vector<wgpu::Bind_گروہ_اندراج> entries = { ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروہ_اندراج(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_سائز = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t کل_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_سائز);
    compute_2d_workgroups(کل_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    واپس ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
ساکن std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           صحیح            گرہ_idx,
                                                           صحیح &          عدد_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  گرہ  = nodes[گرہ_idx];

    اگر (ggml_is_خالی_خانے(گرہ)) {
        واپس std::nullopt;
    }
    اگر ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        واپس std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << گرہ << ", " << ggml_op_name(گرہ->op) << ")");

    ggml_tensor * src0 = گرہ->src[0];
    ggml_tensor * src1 = گرہ->src[1];
    ggml_tensor * src2 = گرہ->src[2];

    انتخاب (گرہ->op) {
            // no-ops
        حالت GGML_OP_NONE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_RESHAPE:
            واپس std::nullopt;
        حالت GGML_OP_DUP:
        حالت GGML_OP_CPY:
        حالت GGML_OP_CONT:
            واپس ggml_webgpu_cpy(ctx, src0, گرہ);
        حالت GGML_OP_سیٹ:
            واپس ggml_webgpu_سیٹ(ctx, src0, src1, گرہ);
        حالت GGML_OP_سیٹ_صفیں:
            واپس ggml_webgpu_سیٹ_صفیں(ctx, src0, src1, گرہ);
        حالت GGML_OP_GET_صفیں:
            واپس ggml_webgpu_get_صفیں(ctx, src0, src1, گرہ);
        حالت GGML_OP_MUL_MAT:
            واپس ggml_webgpu_mul_mat(ctx, src0, src1, گرہ);
        حالت GGML_OP_MUL_MAT_ID:
            واپس ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, گرہ);
        حالت GGML_OP_FLASH_ATTN_EXT:
            واپس ggml_webgpu_flash_attn(ctx, src0, src1, src2, گرہ->src[3], گرہ->src[4], گرہ);
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_SUB:
        حالت GGML_OP_MUL:
        حالت GGML_OP_DIV:
            واپس ggml_webgpu_ثنائی_op(ctx, src0, src1, گرہ);
        حالت GGML_OP_جوڑو_ID:
            واپس ggml_webgpu_جوڑو_id(ctx, src0, src1, src2, گرہ);
        حالت GGML_OP_CONCAT:
            واپس ggml_webgpu_concat(ctx, src0, src1, گرہ);
        حالت GGML_OP_REPEAT:
            واپس ggml_webgpu_repeat(ctx, src0, گرہ);
        حالت GGML_OP_RMS_NORM:
            اگر (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, گرہ_idx)) {
                عدد_encoded_ops        = 2;
                ggml_tensor * mul_گرہ = nodes[گرہ_idx + 1];
                واپس ggml_webgpu_rms_norm_mul(ctx, src0, گرہ, mul_گرہ->src[0], mul_گرہ->src[1], mul_گرہ);
            } ورنہ {
                واپس ggml_webgpu_صف_norm(ctx, src0, گرہ);
            }
        حالت GGML_OP_NORM:
        حالت GGML_OP_L2_NORM:
            واپس ggml_webgpu_صف_norm(ctx, src0, گرہ);
        حالت GGML_OP_ROPE:
            واپس ggml_webgpu_rope(ctx, src0, src1, src2, گرہ);
        حالت GGML_OP_GLU:
            واپس ggml_webgpu_glu(ctx, src0, src1, گرہ);
        حالت GGML_OP_SCALE:
            واپس ggml_webgpu_scale(ctx, src0, گرہ);
        حالت GGML_OP_SOFT_MAX:
            واپس ggml_webgpu_soft_max(ctx, src0, src1, src2, گرہ);
        حالت GGML_OP_UNARY:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_FILL:
        حالت GGML_OP_LOG:
        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_SIN:
        حالت GGML_OP_COS:
        حالت GGML_OP_DIAG:
        حالت GGML_OP_TRI:
            واپس ggml_webgpu_unary_op(ctx, src0, گرہ);
        حالت GGML_OP_SOLVE_TRI:
            واپس ggml_webgpu_solve_tri(ctx, src0, src1, گرہ);
        حالت GGML_OP_SSM_CONV:
            واپس ggml_webgpu_ssm_conv(ctx, src0, src1, گرہ);
        حالت GGML_OP_SSM_جائزہ:
            واپس ggml_webgpu_ssm_جائزہ(ctx, src0, src1, src2, گرہ->src[3], گرہ->src[4], گرہ->src[5], گرہ->src[6],
                                        گرہ);
        حالت GGML_OP_GATED_DELTA_NET:
            اگر (ggml_webgpu_can_fuse_gdn_cache(cgraph, گرہ_idx, عدد_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[گرہ_idx + عدد_encoded_ops - 1]->src[1];
                واپس ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, گرہ->src[3], گرہ->src[4], گرہ->src[5],
                                                   گرہ, dst_fuse);
            } ورنہ {
                واپس ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, گرہ->src[3], گرہ->src[4], گرہ->src[5],
                                                   گرہ, خالی_اشاریہ);
            }
        حالت GGML_OP_PAD:
            واپس ggml_webgpu_pad(ctx, src0, گرہ);
        حالت GGML_OP_ARGMAX:
            واپس ggml_webgpu_argmax(ctx, src0, گرہ);
        حالت GGML_OP_ARGSORT:
        حالت GGML_OP_چوٹی_K:
            // we reuse the same argsort implementation for top_k
            واپس ggml_webgpu_argsort(ctx, src0, گرہ);
        حالت GGML_OP_CUMSUM:
            واپس ggml_webgpu_cumsum(ctx, src0, گرہ);
        حالت GGML_OP_مجموع:
        حالت GGML_OP_مجموع_صفیں:
            واپس ggml_webgpu_مجموع_صفیں(ctx, src0, گرہ);
        حالت GGML_OP_CONV_2D:
            واپس ggml_webgpu_conv_2d(ctx, src0, src1, گرہ);
        حالت GGML_OP_CONV_2D_DW:
            واپس ggml_webgpu_conv_2d_dw(ctx, src0, src1, گرہ);
        حالت GGML_OP_IM2_ستون:
            واپس ggml_webgpu_im2col(ctx, src0, src1, گرہ);
        حالت GGML_OP_UPSCALE:
            واپس ggml_webgpu_upscale(ctx, src0, گرہ);
        طےشدہ:
            واپس std::nullopt;
    }
}

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
ساکن خالی ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ثابت std::vector<std::string> & pipeline_نام,
                                                        uint32_t &                       عدد_inflight_batches) {
    اگر (pipeline_نام.خالی_خانے()) {
        واپس;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Resolve_Query_سیٹ(ctx->profile_timestamp_query_سیٹ, 0, ctx->profile_timestamp_query_گنتی,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_بفر_مقصد_بفر(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_گنتی * حجم(uint64_t));

    wgpu::Command_بفر profile_commands = encoder.اختتام_2();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, عدد_inflight_batches);

    ثابت size_t mapped_سائز = ctx->profile_timestamp_query_گنتی * حجم(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_گنتی == 2 * pipeline_نام.size());

    ggml_backend_webgpu_map_بفر(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::Map_منوال::Read, 0,
                                   mapped_سائز);
    ثابت uint64_t * ts_ڈیٹا = (ثابت uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_سائز);

    برائے (size_t i = 0; i < pipeline_نام.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ثابت دوہرا elapsed_ms = دوہرا(ts_ڈیٹا[2 * i + 1] - ts_ڈیٹا[2 * i]) * 1e-6;
        ctx->shader_gpu_وقت_ms[pipeline_نام[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#اگر_ختم

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
ساکن خالی ggml_backend_webgpu_check_سیٹ_صفیں(webgpu_context & ctx, uint32_t & عدد_inflight_batches) {
#اگر_تعریف_شدہ GGML_WEBGPU_CHECK_سیٹ_صفیں
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_بفر_مقصد_بفر(ctx->سیٹ_صفیں_dev_error_buf, 0, ctx->سیٹ_صفیں_host_error_buf, 0,
                               ctx->سیٹ_صفیں_host_error_buf.Get_سائز());
    wgpu::Command_بفر commands = encoder.اختتام_2();
    ggml_backend_webgpu_submit_commands(ctx, commands, عدد_inflight_batches);
    ggml_backend_webgpu_map_بفر(ctx->global_ctx, ctx->سیٹ_صفیں_host_error_buf, wgpu::Map_منوال::Read, 0,
                                   ctx->سیٹ_صفیں_host_error_buf.Get_سائز());
    ثابت uint32_t * error_ڈیٹا = (ثابت uint32_t *) ctx->سیٹ_صفیں_host_error_buf.GetConstMappedRange();
    اگر (*error_ڈیٹا) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->سیٹ_صفیں_host_error_buf.Unmap();
#ورنہ
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(عدد_inflight_batches);
#اگر_ختم
}

ساکن ggml_حالت ggml_backend_webgpu_گراف_compute(ggml_backend_t backend, ساخت ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_کل_آغاز(گراف_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t عدد_batched_kernels  = 0;
    uint32_t عدد_inflight_batches = 0;
    منطقی     شامل_ہے_سیٹ_صفیں    = جھوٹ;
    صحیح      عدد_encoded_ops      = 1;
    صحیح      گرہ_idx             = 0;

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_گنتی = 0;
    std::vector<std::string> profile_pipeline_نام;
#اگر_ختم

    ctx->فعال_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    اگر (ctx->batch_compute_passes) {
        ctx->فعال_compute_pass = ctx->فعال_command_encoder.BeginComputePass();
    }

    جب (گرہ_idx < cgraph->n_nodes) {
        اگر (cgraph->nodes[گرہ_idx]->op == GGML_OP_سیٹ_صفیں) {
            شامل_ہے_سیٹ_صفیں = سچ;
        }
        اگر (خودکار cmd = ggml_webgpu_encode(ctx, cgraph, گرہ_idx, عدد_encoded_ops)) {
            commands.push_back(*cmd);
            عدد_batched_kernels += cmd.value().عدد_kernels;
#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_نام.insert(profile_pipeline_نام.end(), cmd->pipeline_نام.begin(),
                                          cmd->pipeline_نام.end());
#اگر_ختم
        }

        اگر (عدد_batched_kernels >= ctx->global_ctx->command_submit_batch_سائز) {
            اگر (ctx->فعال_compute_pass) {
                ctx->فعال_compute_pass.انتہا_2();
            }
            عدد_batched_kernels                = 0;
            wgpu::Command_بفر batch_commands = ctx->فعال_command_encoder.اختتام_2();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, عدد_inflight_batches);

            // reset state for next batch
            ctx->فعال_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            اگر (ctx->batch_compute_passes) {
                ctx->فعال_compute_pass = ctx->فعال_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            اگر (ctx->profile_timestamp_query_گنتی + 2 * ctx->global_ctx->command_submit_batch_سائز >=
                WEBGPU_MAX_PROFILE_QUERY_گنتی) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_نام, عدد_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_گنتی = 0;
                profile_pipeline_نام.clear();
            }
#اگر_ختم
        }

        گرہ_idx += عدد_encoded_ops;
        عدد_encoded_ops = 1;
    }

    اگر (ctx->فعال_compute_pass) {
        ctx->فعال_compute_pass.انتہا_2();
        ctx->فعال_compute_pass = خالی_اشاریہ;
    }

    اگر (عدد_batched_kernels > 0) {
        wgpu::Command_بفر batch_commands = ctx->فعال_command_encoder.اختتام_2();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, عدد_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->فعال_command_encoder = خالی_اشاریہ;

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_نام, عدد_inflight_batches);
#اگر_ختم

    اگر (شامل_ہے_سیٹ_صفیں) {
        ggml_backend_webgpu_check_سیٹ_صفیں(ctx, عدد_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_کل_انتہا(گراف_compute, ctx->global_ctx);
    واپس GGML_حالت_SUCCESS;
}

ساخت ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    منطقی                  recorded = جھوٹ;
};

ساکن ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    خودکار * event_ctx      = نیا ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    خودکار * event   = نیا ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    واپس event;
}

ساکن خالی ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_غیر_مستعمل(dev);
    حذف static_cast<ggml_backend_webgpu_event_context *>(event->context);
    حذف event;
}

ساکن خالی ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_غیر_مستعمل(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    اگر (!event_ctx->recorded) {
        واپس;
    }
    wgpu::Wait_حالت حالت_2 =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    اگر (حالت_2 == wgpu::Wait_حالت::Timed_برآمد) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = جھوٹ;
}

ساکن خالی ggml_backend_webgpu_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.On_Submitted_Work_مکمل(
        wgpu::Callback_منوال::AllowSpontaneous, [](wgpu::قطار_Work_مکمل_حالت, wgpu::StringView) {});
    event_ctx->recorded = سچ;
}

ساکن خالی ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_غیر_مستعمل(backend);
    ggml_backend_webgpu_device_event_synchronize(خالی_اشاریہ, event);
}

ساکن خالی ggml_backend_webgpu_سیٹ_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ثابت خالی *   ڈیٹا,
                                                 size_t         انحراف,
                                                 size_t         size) {
    GGML_غیر_مستعمل(backend);
    خودکار * buf_ctx      = (ggml_backend_webgpu_بفر_context *) tensor->بفر->context;
    size_t کل_انحراف = ggml_webgpu_tensor_انحراف(tensor) + انحراف;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_بفر(buf_ctx->بفر, کل_انحراف, ڈیٹا, (size / 4) * 4);

    اگر (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t بقایا_سائز = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        برائے (size_t i = 0; i < بقایا_سائز; i++) {
            ((uint8_t *) &val32)[i] = ((ثابت uint8_t *) ڈیٹا)[size - بقایا_سائز + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, val32,
                                          کل_انحراف + (size - بقایا_سائز), بقایا_سائز);
    }
}

ساکن خالی ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_قطار(backend_ctx->webgpu_ctx->global_ctx);
}

ساکن ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_سیٹ_tensor_async,
    /* .get_tensor_async        = */ تہی,
    /* .set_tensor_2d_async     = */ تہی,
    /* .get_tensor_2d_async     = */ تہی,
    /* .cpy_tensor_async        = */ تہی,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_webgpu_گراف_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ تہی,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

ساکن خالی ggml_backend_webgpu_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_webgpu_بفر_context * ctx = static_cast<ggml_backend_webgpu_بفر_context *>(بفر->context);
    اگر (ctx != خالی_اشاریہ && ctx->بفر != خالی_اشاریہ) {
        ctx->بفر.Destroy();
        حذف ctx;
    }
}

// Returns the "fake" base pointer.
ساکن خالی * ggml_backend_webgpu_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    GGML_غیر_مستعمل(بفر);
    واپس webgpu_ptr_بنیاد;
}

ساکن خالی ggml_backend_webgpu_بفر_memset_tensor(ggml_backend_بفر_t بفر,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                انحراف,
                                                     size_t                size) {
    اگر (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        واپس;
    }

    WEBGPU_CPU_PROFILE_کل_آغاز(memset_tensor);

    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << انحراف << ", " << size << ")");

    size_t کل_انحراف = ggml_webgpu_tensor_انحراف(tensor) + انحراف;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, val32, کل_انحراف, size);
    WEBGPU_CPU_PROFILE_کل_انتہا(memset_tensor, buf_ctx->global_ctx);
}

ساکن خالی ggml_backend_webgpu_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                  ggml_tensor *         tensor,
                                                  ثابت خالی *          ڈیٹا,
                                                  size_t                انحراف,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_کل_آغاز(سیٹ_tensor);
    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << ڈیٹا
                                                              << ", " << انحراف << ", " << size << ")");

    size_t کل_انحراف = ggml_webgpu_tensor_انحراف(tensor) + انحراف;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    اگر (کل_انحراف % 4 != 0) {
        size_t lane = کل_انحراف % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t سر = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t سر_val = 0;
        برائے (size_t i = 0; i < سر; i++) {
            ((uint8_t *) &سر_val)[lane + i] = ((ثابت uint8_t *) ڈیٹا)[i];
        }
        ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, سر_val, کل_انحراف, سر);

        کل_انحراف += سر;
        size -= سر;
        ڈیٹا = (ثابت uint8_t *) ڈیٹا + سر;
    }

    اگر (size > 0) {
        buf_ctx->global_ctx->queue.Write_بفر(buf_ctx->بفر, کل_انحراف, ڈیٹا, (size / 4) * 4);
    }

    اگر (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t بقایا_سائز = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        برائے (size_t i = 0; i < بقایا_سائز; i++) {
            ((uint8_t *) &val32)[i] = ((ثابت uint8_t *) ڈیٹا)[size - بقایا_سائز + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, val32,
                                          کل_انحراف + (size - بقایا_سائز), بقایا_سائز);
    }
    WEBGPU_CPU_PROFILE_کل_انتہا(سیٹ_tensor, buf_ctx->global_ctx);
}

ساکن خالی ggml_backend_webgpu_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                  ثابت ggml_tensor *   tensor,
                                                  خالی *                ڈیٹا,
                                                  size_t                انحراف,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_کل_آغاز(get_tensor);
    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << ڈیٹا
                                                              << ", " << انحراف << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t کل_انحراف = ggml_webgpu_tensor_انحراف(tensor) + انحراف;

    size_t local_انحراف = کل_انحراف % 4;
    اگر (local_انحراف != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        کل_انحراف -= local_انحراف;
    }

    size_t final_سائز = size + local_انحراف;
    اگر (final_سائز % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_سائز += 4 - (final_سائز % 4);
    }

    std::lock_guard<std::recursive_mutex> تالا(buf_ctx->global_ctx->mutex);

    اگر (buf_ctx->global_ctx->get_tensor_staging_buf == خالی_اشاریہ ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_سائز() < final_سائز) {
        // Create a new staging buffer if it doesn't exist or is too small
        اگر (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_بفر(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_سائز,
                                  wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_بفر_مقصد_بفر(buf_ctx->بفر, کل_انحراف, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_سائز);
    wgpu::Command_بفر commands = encoder.اختتام_2();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_بفر(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::Map_منوال::Read, 0, final_سائز);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ثابت خالی * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_سائز);

    // Copy the data from the mapped range to the output buffer
    std::حافظہ_نقل(ڈیٹا, (ثابت خالی *) ((ثابت حرف *) mapped_range + local_انحراف), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_کل_انتہا(get_tensor, buf_ctx->global_ctx);
}

ساکن خالی ggml_backend_webgpu_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << بفر << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_کل_آغاز(clear);
    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;
    ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, value, 0, بفر->size);
    WEBGPU_CPU_PROFILE_کل_انتہا(clear, buf_ctx->global_ctx);
}

ساکن ggml_backend_بفر_i ggml_backend_webgpu_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_webgpu_بفر_get_بنیاد,
    /* .init_tensor     = */ تہی,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ تہی,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_بفر_clear,
    /* .reset           = */ تہی,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

ساکن ثابت حرف * ggml_backend_webgpu_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    واپس ctx->device_name.c_str();
}

ساکن ggml_backend_بفر_t ggml_backend_webgpu_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                          size_t                     size) {
    ساکن std::atomic<صحیح> بفر_گنتی;
    صحیح                     بفر_id = بفر_گنتی++;
    std::string             buf_name  = "tensor_buf" + std::to_string(بفر_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << بفر_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::بفر_2                         buf;
    ggml_webgpu_create_بفر(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::بفر_Usage::Storage | wgpu::بفر_Usage::CopySrc | wgpu::بفر_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_بفر_context * buf_ctx =
        نیا ggml_backend_webgpu_بفر_context(buf, buf_name, ctx->webgpu_global_ctx);

    واپس ggml_backend_بفر_init(buft, ggml_backend_webgpu_بفر_interface, buf_ctx, size);
}

ساکن size_t ggml_backend_webgpu_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    واپس dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
ساکن size_t ggml_backend_webgpu_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    واپس dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز;
}

ساکن size_t ggml_backend_webgpu_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft,
                                                             ثابت ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    انتخاب (tensor->op) {
        حالت GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            توڑو;
        حالت GGML_OP_چوٹی_K:
            {
                ثابت ggml_tensor * src0 = tensor->src[0];
                اگر (src0) {
                    ثابت size_t بھرا = حجم(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        بھرا * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            توڑو;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                ثابت ggml_tensor * Q            = tensor->src[0];
                ثابت ggml_tensor * K            = tensor->src[1];
                ثابت ggml_tensor * V            = tensor->src[2];
                ثابت ggml_tensor * نقاب         = tensor->src[3];
                ثابت خودکار &        capabilities = ctx->webgpu_global_ctx->capabilities;
                اگر (ggml_webgpu_flash_attn_use_vec_راستہ(ctx->webgpu_global_ctx, Q, K, V)) {
                    ثابت منطقی kv_براہ_راست =
                        ggml_webgpu_flash_attn_k_براہ_راست(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی) ||
                        ggml_webgpu_flash_attn_v_براہ_راست(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی);
                    ثابت uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_سائز, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        نقاب != خالی_اشاریہ, kv_براہ_راست);

                    ثابت uint32_t vec_nwg_cap = capabilities.min_subgroup_سائز;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ثابت size_t   align = capabilities.limits.min_Storage_بفر_انحراف_Alignment;
                    ثابت uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    اگر (nwg > 1u) {
                        ثابت uint64_t tmp_ڈیٹا_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ثابت uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ثابت size_t   tmp_سائز_بائٹیں = ROUNDUP_POW2((tmp_ڈیٹا_elems + tmp_stats_elems) * حجم(اعشاری),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_سائز_بائٹیں + align;
                    } ورنہ {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    اگر (نقاب != خالی_اشاریہ) {
                        ثابت uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ثابت uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ثابت uint32_t stride_mask3    = (uint32_t) (نقاب->nb[3] / ggml_type_سائز(نقاب->type));
                        ثابت uint32_t blk_batch_گنتی = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ثابت uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_گنتی;
                        ثابت size_t   blk_سائز_بائٹیں =
                            ROUNDUP_POW2(blk_elems * حجم(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_سائز_بائٹیں + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            توڑو;
        حالت GGML_OP_MUL_MAT:
            {
                ثابت ggml_tensor * src0 = tensor->src[0];
                ثابت ggml_tensor * src1 = tensor->src[1];
                منطقی                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_حاصل_ضرب,
                                             ctx->webgpu_global_ctx->vendor);
                اگر (use_mmvq) {
                    ثابت size_t q8_src1_سائز = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_سائز +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            توڑو;
        حالت GGML_OP_MUL_MAT_ID:
            {
                ثابت ggml_tensor * src0 = tensor->src[0];
                ثابت ggml_tensor * src1 = tensor->src[1];
                اگر (src0 && src1) {
                    ثابت size_t ids_cache_سائز = حجم(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    ثابت size_t ids_tiles_سائز = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_سائز + ids_tiles_سائز +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_انحراف_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            توڑو;
        طےشدہ:
            توڑو;
    }
    واپس res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

ساکن ثابت حرف * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    واپس ctx->device_name.c_str();
}

ساکن ثابت حرف * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    واپس ctx->device_desc.c_str();
}

ساکن خالی ggml_backend_webgpu_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_بفر_سائز = ctx->webgpu_global_ctx->capabilities.limits.max_بفر_سائز_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#اگر UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_سائز = static_cast<uint64_t>(UINTPTR_MAX);
    اگر (max_بفر_سائز > max_ptr_سائز) {
        max_بفر_سائز = max_ptr_سائز;
    }
#اگر_ختم
    *آزاد  = static_cast<size_t>(max_بفر_سائز);
    *کل = static_cast<size_t>(max_بفر_سائز);
}

ساکن شمار ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_غیر_مستعمل(dev);
    واپس GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساکن خالی ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, ساخت ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                 = */ جھوٹ,
        /* .host_buffer           = */ جھوٹ,
        /* .buffer_from_host_ptr  = */ جھوٹ,
        /* .events                = */ جھوٹ,
        /* .mmap_support          = */ سچ,
    };
}

ساکن ggml_guid_t ggml_backend_webgpu_guid(خالی) {
    ساکن ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    واپس &guid;
}

ساکن خالی ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_دھاگے = ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup *
                         ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_بائٹیں_فی_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_بفر_Binding_سائز, max_دھاگے);
    std::vector<wgpu::Constant_اندراج> constants(2);
    constants[0].کلید     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    constants[1].کلید     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_بائٹیں_فی_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

ساکن خالی ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#اگر !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ثابت حرف * ثابت          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_گنتی = 2;
    options.اگلا_اندر_Chain                   = &adapterTogglesDesc;
#اگر_ختم

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::Callback_منوال::AllowSpontaneous,
            [&adapter](wgpu::Request_Adapter_حالت حالت_2, wgpu::Adapter _adapter, wgpu::StringView پیغام) {
                اگر (حالت_2 != wgpu::Request_Adapter_حالت::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(پیغام).c_str());
                    واپس;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

ساکن خالی create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != خالی_اشاریہ);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#اگر !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::Adapter_Properties_Subgroup_میٹرکس_Configs subgroup_میٹرکس_configs{};
    اگر (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_میٹرکس)) {
        info.اگلا_اندر_Chain = &subgroup_میٹرکس_configs;
    }
#اگر_ختم
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_سائز = ggml_backend_webgpu_get_command_submit_batch_سائز();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_حاصل_ضرب = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8_Integer_Dot_حاصل_ضرب);

    منطقی valid_subgroup_میٹرکس_config = جھوٹ;
#اگر !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    اگر (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_میٹرکس)) {
        برائے (size_t i = 0; i < subgroup_میٹرکس_configs.config_گنتی; i++) {
            ثابت wgpu::Subgroup_میٹرکس_Config config = subgroup_میٹرکس_configs.configs[i];
            اگر (config.componentType == wgpu::Subgroup_میٹرکس_Component_Type::F16 &&
                config.نتیجہ_Component_Type == wgpu::Subgroup_میٹرکس_Component_Type::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_میٹرکس_config                  = سچ;
                توڑو;
            }
        }
    }
#اگر_ختم
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_میٹرکس = valid_subgroup_میٹرکس_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_سائز = info.subgroup_Min_سائز;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_سائز = info.subgroup_Max_سائز;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#اگر !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    اگر (ctx->webgpu_global_ctx->capabilities.supports_subgroup_میٹرکس) {
        required_features.push_back(wgpu::FeatureName::Chromium_Experimental_Subgroup_میٹرکس);
    }
#اگر_ختم

    اگر (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#اگر_ختم

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.ڈیٹا();
    dev_desc.required_Feature_گنتی = required_features.size();
    dev_desc.سیٹ_Device_گمشدہ_Callback(
        wgpu::Callback_منوال::AllowSpontaneous,
        [](ثابت wgpu::Device & device, wgpu::Device_گمشدہ_Reason reason, wgpu::StringView پیغام) {
            اگر (reason == wgpu::Device_گمشدہ_Reason::Destroyed) {
                واپس;
            }
            GGML_غیر_مستعمل(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<صحیح>(reason),
                           std::string(پیغام).c_str());
        });
    dev_desc.سیٹ_Uncaptured_Error_Callback(
        [](ثابت wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView پیغام) {
            GGML_غیر_مستعمل(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<صحیح>(reason),
                       std::string(پیغام).c_str());
        });

#اگر !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    ثابت حرف * ثابت          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    ثابت حرف * ثابت          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_گنتی  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_گنتی = 1;

    dev_desc.اگلا_اندر_Chain = &deviceTogglesDesc;
#اگر_ختم

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::Callback_منوال::AllowSpontaneous,
            [ctx](wgpu::Request_Device_حالت حالت_2, wgpu::Device device, wgpu::StringView پیغام) {
                اگر (حالت_2 != wgpu::Request_Device_حالت::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(پیغام).c_str());
                    واپس;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != خالی_اشاریہ);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_بفر(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_سائز_بائٹیں, wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.Get_قطار();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

ساکن webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_سائز_بائٹیں,
        webgpu_ctx->global_ctx->command_submit_batch_سائز + WEBGPU_عدد_PARAM_خانہ_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_یکساں_بفر_انحراف_Alignment);
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->سیٹ_صفیں_dev_error_buf,
                              WEBGPU_سیٹ_صفیں_ERROR_BUF_سائز_بائٹیں,
                              wgpu::بفر_Usage::Storage | wgpu::بفر_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->سیٹ_صفیں_host_error_buf,
                              WEBGPU_سیٹ_صفیں_ERROR_BUF_سائز_بائٹیں,
                              wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "set_rows_host_error_buf");

#اگر_تعریف_شدہ GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = جھوٹ;
    ggml_webgpu_create_بفر(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_سائز_بائٹیں,
        wgpu::بفر_Usage::QueryResolve | wgpu::بفر_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_سائز_بائٹیں,
                              wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::Query_سیٹ_Descriptor query_سیٹ_desc = {};
    query_سیٹ_desc.type                     = wgpu::QueryType::Timestamp;
    query_سیٹ_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_گنتی;
    webgpu_ctx->profile_timestamp_query_سیٹ = webgpu_ctx->global_ctx->device.Create_Query_سیٹ(&query_سیٹ_desc);
#اگر_ختم

#اگر_تعریف_شدہ GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * حجم(uint32_t),
                              wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * حجم(uint32_t),
                              wgpu::بفر_Usage::Storage | wgpu::بفر_Usage::CopySrc, "debug_dev_buf");
#اگر_ختم
    واپس webgpu_ctx;
}

ساکن ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_غیر_مستعمل(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    خودکار * backend_ctx      = نیا ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    خودکار * backend = نیا ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    واپس backend;
}

ساکن ggml_backend_بفر_type_t ggml_backend_webgpu_device_get_بفر_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    ساکن ساخت ggml_backend_بفر_type ggml_backend_webgpu_بفر_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_بفر_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_بفر_type_alloc_بفر,
                        /* .alloc_buffer_n      = */ تہی,
                        /* .get_alignment       = */ ggml_backend_webgpu_بفر_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_بفر_type_get_max_سائز,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_بفر_type_get_alloc_سائز,
                        /* .get_alloc_size_n    = */ تہی,
                        /* .is_host             = */ تہی,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ تہی
    };

    واپس &ggml_backend_webgpu_بفر_type;
}

ساکن منطقی ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    GGML_غیر_مستعمل(dev);
    واپس buft->iface.get_name == ggml_backend_webgpu_بفر_type_get_name;
}

ساکن منطقی ggml_webgpu_supported_qtype(ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ2_S:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ3_S:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_NVFP4:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

ساکن منطقی ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    اگر (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز ||
        (src0 != خالی_اشاریہ &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز) ||
        (src1 != خالی_اشاریہ &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز)) {
        واپس جھوٹ;
    }

    منطقی supports_op = جھوٹ;
    انتخاب (op->op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_RESHAPE:
            supports_op = سچ;
            توڑو;
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_SUB:
        حالت GGML_OP_MUL:
        حالت GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            توڑو;
        حالت GGML_OP_جوڑو_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            توڑو;
        حالت GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            توڑو;
        حالت GGML_OP_DUP:
        حالت GGML_OP_CPY:
        حالت GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            توڑو;
        حالت GGML_OP_سیٹ:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            توڑو;
        حالت GGML_OP_سیٹ_صفیں:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            توڑو;
        حالت GGML_OP_GET_صفیں:
            اگر (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } ورنہ اگر (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            توڑو;
        حالت GGML_OP_MUL_MAT:
            {
                انتخاب (src1->type) {
                    حالت GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        توڑو;
                    حالت GGML_TYPE_F32:
                        انتخاب (src0->type) {
                            حالت GGML_TYPE_F32:
                            حالت GGML_TYPE_F16:
                            حالت GGML_TYPE_BF16:
                            حالت GGML_TYPE_Q1_0:
                            حالت GGML_TYPE_Q2_0:
                            حالت GGML_TYPE_Q4_0:
                            حالت GGML_TYPE_Q4_1:
                            حالت GGML_TYPE_Q5_0:
                            حالت GGML_TYPE_Q5_1:
                            حالت GGML_TYPE_Q8_0:
                            حالت GGML_TYPE_Q2_K:
                            حالت GGML_TYPE_Q3_K:
                            حالت GGML_TYPE_Q4_K:
                            حالت GGML_TYPE_Q5_K:
                            حالت GGML_TYPE_Q6_K:
                            حالت GGML_TYPE_IQ2_XXS:
                            حالت GGML_TYPE_IQ2_XS:
                            حالت GGML_TYPE_IQ2_S:
                            حالت GGML_TYPE_IQ3_XXS:
                            حالت GGML_TYPE_IQ3_S:
                            حالت GGML_TYPE_IQ1_S:
                            حالت GGML_TYPE_IQ1_M:
                            حالت GGML_TYPE_IQ4_NL:
                            حالت GGML_TYPE_IQ4_XS:
                            حالت GGML_TYPE_MXFP4:
                            حالت GGML_TYPE_NVFP4:
                                supports_op = سچ;
                                توڑو;
                            طےشدہ:
                                توڑو;
                        }
                    طےشدہ:
                        توڑو;
                }
                توڑو;
            }
        حالت GGML_OP_MUL_MAT_ID:
            انتخاب (src1->type) {
                حالت GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    توڑو;
                حالت GGML_TYPE_F32:
                    انتخاب (src0->type) {
                        حالت GGML_TYPE_F32:
                        حالت GGML_TYPE_F16:
                        حالت GGML_TYPE_BF16:
                        حالت GGML_TYPE_Q1_0:
                        حالت GGML_TYPE_Q4_0:
                        حالت GGML_TYPE_Q4_1:
                        حالت GGML_TYPE_Q5_0:
                        حالت GGML_TYPE_Q5_1:
                        حالت GGML_TYPE_Q8_0:
                        حالت GGML_TYPE_Q2_K:
                        حالت GGML_TYPE_Q3_K:
                        حالت GGML_TYPE_Q4_K:
                        حالت GGML_TYPE_Q5_K:
                        حالت GGML_TYPE_Q6_K:
                        حالت GGML_TYPE_IQ1_S:
                        حالت GGML_TYPE_IQ1_M:
                        حالت GGML_TYPE_IQ2_XXS:
                        حالت GGML_TYPE_IQ2_XS:
                        حالت GGML_TYPE_IQ2_S:
                        حالت GGML_TYPE_IQ3_XXS:
                        حالت GGML_TYPE_IQ3_S:
                        حالت GGML_TYPE_IQ4_NL:
                        حالت GGML_TYPE_IQ4_XS:
                        حالت GGML_TYPE_MXFP4:
                        حالت GGML_TYPE_NVFP4:
                            supports_op = سچ;
                            توڑو;
                        طےشدہ:
                            توڑو;
                    }
                    توڑو;
                طےشدہ:
                    توڑو;
            }
            اگر (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                اگر (!supports_op) {
                    توڑو;
                }
                اگر (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = جھوٹ;
                    توڑو;
                }
                ثابت خودکار & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ثابت size_t storage_انحراف_alignment = capabilities.limits.min_Storage_بفر_انحراف_Alignment;

                // subgroup matrix path requirements
                ثابت منطقی use_subgroup_میٹرکس = ggml_webgpu_flash_attn_can_use_subgroup_میٹرکس_راستہ(
                    capabilities.supports_subgroup_میٹرکس, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ثابت منطقی float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_انحراف_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_انحراف_alignment));
                ثابت uint32_t k_tile_سر_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی :
                                                       (uint32_t) ggml_blck_سائز(src1->type);
                ثابت uint32_t v_tile_سر_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_چوڑائی :
                                                       (uint32_t) ggml_blck_سائز(src2->type);
                ثابت منطقی     tile_kv_سر_dims_aligned =
                    src0->ne[0] % k_tile_سر_align == 0 && src2->ne[0] % v_tile_سر_align == 0;
                ثابت منطقی tile_can_dispatch_all_q_صفیں =
                    capabilities.limits.max_Compute_Invocations_فی_Workgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_سائز;
                ثابت منطقی use_tile = !use_subgroup_میٹرکس && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_سر_dims_aligned && tile_can_dispatch_all_q_صفیں;

                اگر (!use_subgroup_میٹرکس && !use_tile) {
                    supports_op = جھوٹ;
                    توڑو;
                }
                ثابت uint32_t q_tile =
                    use_subgroup_میٹرکس ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ثابت uint32_t kv_granularity = use_subgroup_میٹرکس ? capabilities.sg_mat_n : 1u;
                ثابت منطقی     kv_براہ_راست = use_subgroup_میٹرکس ?
                                               ggml_webgpu_flash_attn_k_براہ_راست(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_براہ_راست(src0, src2, capabilities.sg_mat_k) :
                                               جھوٹ;
                ثابت uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_سائز, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != خالی_اشاریہ, kv_براہ_راست);
                supports_op = max_kv_tile > 0;
                توڑو;
            }
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_NORM:
        حالت GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_صفیں(src0);
            توڑو;
        حالت GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            توڑو;
        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(op)) {
                حالت GGML_GLU_OP_REGLU:
                حالت GGML_GLU_OP_GEGLU:
                حالت GGML_GLU_OP_SWIGLU:
                حالت GGML_GLU_OP_GEGLU_ERF:
                حالت GGML_GLU_OP_GEGLU_تیز:
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    توڑو;
                حالت GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    توڑو;
                طےشدہ:
                    توڑو;
            }
            توڑو;
        حالت GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_UNARY:
            {
                ثابت ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                انتخاب (UNARY_OP) {
                    حالت GGML_UNARY_OP_ABS:
                    حالت GGML_UNARY_OP_SGN:
                    حالت GGML_UNARY_OP_NEG:
                    حالت GGML_UNARY_OP_قدم:
                    حالت GGML_UNARY_OP_TANH:
                    حالت GGML_UNARY_OP_ELU:
                    حالت GGML_UNARY_OP_RELU:
                    حالت GGML_UNARY_OP_SIGMOID:
                    حالت GGML_UNARY_OP_GELU:
                    حالت GGML_UNARY_OP_GELU_تیز:
                    حالت GGML_UNARY_OP_SILU:
                    حالت GGML_UNARY_OP_HARDSWISH:
                    حالت GGML_UNARY_OP_HARDSIGMOID:
                    حالت GGML_UNARY_OP_EXP:
                    حالت GGML_UNARY_OP_GELU_ERF:
                    حالت GGML_UNARY_OP_SOFTPLUS:
                    حالت GGML_UNARY_OP_EXPM1:
                    حالت GGML_UNARY_OP_FLOOR:
                    حالت GGML_UNARY_OP_CEIL:
                    حالت GGML_UNARY_OP_باری:
                    حالت GGML_UNARY_OP_TRUNC:
                    حالت GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        توڑو;
                    طےشدہ:
                        توڑو;
                }
            }
            توڑو;
        حالت GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            توڑو;
        حالت GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            توڑو;
        حالت GGML_OP_IM2_ستون:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            توڑو;
        حالت GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_SSM_جائزہ:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
            توڑو;
        حالت GGML_OP_GATED_DELTA_NET:
            {
                ثابت uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
            }
            توڑو;
        حالت GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            توڑو;
        حالت GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            توڑو;
        حالت GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            توڑو;
        حالت GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            توڑو;
        حالت GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            توڑو;
        حالت GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            توڑو;
        حالت GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            توڑو;
        حالت GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            توڑو;
        حالت GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_صفیں(src0);
            توڑو;
        حالت GGML_OP_چوٹی_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_صفیں(src0);
            توڑو;
        حالت GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            توڑو;
        حالت GGML_OP_مجموع:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            توڑو;
        حالت GGML_OP_مجموع_صفیں:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_صفیں(src0);
            توڑو;
        حالت GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            توڑو;
        طےشدہ:
            توڑو;
    }
    اگر (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز ||
        (src0 != خالی_اشاریہ &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز) ||
        (src1 != خالی_اشاریہ &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز) ||
        (src2 != خالی_اشاریہ &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_سائز)) {
        supports_op = جھوٹ;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    اگر (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } ورنہ {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    واپس supports_op;
}

ساکن ساخت ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_بفر_type,
    /* .get_host_buffer_type = */ تہی,
    /* .buffer_from_host_ptr = */ تہی,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ تہی,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

ساکن ثابت حرف * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    واپس ctx->name;
}

ساکن size_t ggml_backend_webgpu_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    واپس ctx->device_گنتی;
}

// Only one device is supported for now
ساکن ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    GGML_ASSERT(اشاریہ == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_کل_آغاز(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    ساکن ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    ساکن ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_کل_انتہا(reg_get_device, reg_ctx->webgpu_global_ctx);
    واپس &device;
}

ساکن ثابت ساخت ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ تہی,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    ساکن ggml_backend_webgpu_reg_context * ctx = نیا ggml_backend_webgpu_reg_context();

    ساکن ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_گنتی = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    اگر (ctx->webgpu_global_ctx != خالی_اشاریہ && ctx->webgpu_global_ctx->instance != خالی_اشاریہ) {
        واپس &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.ڈیٹا();
    instance_descriptor.required_Feature_گنتی                 = instance_features.size();

#اگر !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ثابت حرف * ثابت          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_گنتی = 1;
    instance_descriptor.اگلا_اندر_Chain        = &instanceTogglesDesc;
#اگر_ختم

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(نیا webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    اگر (ctx->webgpu_global_ctx->instance != خالی_اشاریہ) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    اگر (adapter != خالی_اشاریہ && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#اگر !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#اگر_ختم
    ) {
        ctx->device_گنتی = 1;
    }

    واپس &reg;
}

ggml_backend_t ggml_backend_webgpu_init(خالی) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    اگر (ggml_backend_reg_dev_گنتی(reg) == 0) {
        واپس خالی_اشاریہ;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    واپس ggml_backend_webgpu_backend_init(dev, خالی_اشاریہ);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
