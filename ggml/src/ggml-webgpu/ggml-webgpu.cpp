/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#ضمن "ggml-webgpu.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-webgpu-shader-lib.hpp"
#ضمن "ggml.h"

#که_تعریف_شوی __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#د_که_پای

#ضمن <webgpu/webgpu_cpp.h>

#ضمن <atomic>
#ضمن <cstdint>
#ضمن <cstring>
#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#د_که_پای
#إذا defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#د_که_پای
#ضمن <memory>
#ضمن <mutex>
#ضمن <optional>
#ضمن <string>
#ضمن <utility>
#ضمن <vector>

#تعریف_کړه ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#تعریف_کړه CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
ساكن مضمن فراغ compute_2d_workgroups(uint32_t ټول_wg, uint32_t max_فی_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(ټول_wg, max_فی_dim));
    wg_x = CEIL_DIV(ټول_wg, wg_y);
}

ساكن مضمن uint32_t ggml_webgpu_u32_from_f32(عائم value) {
    uint32_t بټونه;
    حافظه_نقل(&بټونه, &value, حجم(بټونه));
    أرجع بټونه;
}

#که_تعریف_شوی GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(پیغام_2)  std::وتنه << پیغام_2 << std::کرښه_پای
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#وإلا
#    define WEBGPU_LOG_DEBUG(پیغام_2) ((فراغ) 0)
#د_که_پای  // GGML_WEBGPU_DEBUG

#که_تعریف_شوی GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_ټول_پیل(id) خپلکاری cpu_ټول_پیل##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_ټول_پای_ټکی(id, ctx)                                                         \
        خپلکاری   cpu_ټول_پای_ټکی##id = std::chrono::high_resolution_clock::now();                            \
        مزدوج cpu_ټول_وخت##id =                                                                      \
            std::chrono::duration<مزدوج, std::milli>(cpu_ټول_پای_ټکی##id - cpu_ټول_پیل##id).count(); \
        (ctx)->cpu_وخت_ms[#id] += cpu_ټول_وخت##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_پیل(id) خپلکاری cpu_detail_پیل##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_پای_ټکی(id, ctx)                                                          \
        خپلکاری   cpu_detail_پای_ټکی##id = std::chrono::high_resolution_clock::now();                             \
        مزدوج cpu_detail_وخت##id =                                                                       \
            std::chrono::duration<مزدوج, std::milli>(cpu_detail_پای_ټکی##id - cpu_detail_پیل##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_وخت##id;
#وإلا
#    define WEBGPU_CPU_PROFILE_ټول_پیل(id)
#    define WEBGPU_CPU_PROFILE_ټول_پای_ټکی(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_پیل(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_پای_ټکی(id, ctx)
#د_که_پای  // GGML_WEBGPU_CPU_PROFILE

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_شمېر        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_اندازه_بایټونه (WEBGPU_MAX_PROFILE_QUERY_شمېر * حجم(uint64_t))
#د_که_پای

/* Constants */

#تعریف_کړه WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_اندازه 64u
#تعریف_کړه WEBGPU_عدد_PARAM_خونه_SAFETY_MARGIN      10u
#تعریف_کړه WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#تعریف_کړه WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#تعریف_کړه WEBGPU_PARAMS_BUF_اندازه_بایټونه             128  // enough for 32 parameters
#تعریف_کړه WEBGPU_سېټ_لیکې_ERROR_BUF_اندازه_بایټونه     4
#تعریف_کړه WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
ساكن فراغ * ثابت webgpu_ptr_بنسټ = (فراغ *) (uintptr_t) 0x1000;  // NOLINT

ساكن size_t ggml_webgpu_tensor_دننه_واټن(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * بنسټ_tensor = tensor->view_src ? tensor->view_src : tensor;
    أرجع (size_t) ((uintptr_t) بنسټ_tensor->معلومات - (uintptr_t) webgpu_ptr_بنسټ) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
ساكن فراغ ggml_webgpu_create_بفر(wgpu::Device &    device,
                                      wgpu::بفر_2 &    بفر,
                                      size_t            size,
                                      wgpu::بفر_Usage usage,
                                      ثابت حرف *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
بنية webgpu_param_arena {
    wgpu::بفر_2 بفر;
    size_t       خونه_stride = 0;
    size_t       خونه_اندازه   = 0;
    uint32_t     خونه_شمېر  = 0;
    uint32_t     بل_خونه   = 0;

    فراغ init(wgpu::Device device, size_t خونه_اندازه, uint32_t خونه_شمېر, size_t alignment) {
        هذا->خونه_stride = ROUNDUP_POW2(خونه_اندازه, alignment);
        هذا->خونه_اندازه   = خونه_اندازه;
        هذا->خونه_شمېر  = خونه_شمېر;
        هذا->بل_خونه   = 0;

        ggml_webgpu_create_بفر(device, بفر, هذا->خونه_stride * خونه_شمېر,
                                  wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_خونه(size_t size) {
        GGML_ASSERT(size <= خونه_اندازه);
        إذا (بل_خونه >= خونه_شمېر) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        أرجع خونه_stride * بل_خونه++;
    }

    فراغ reset() { بل_خونه = 0; }

    فراغ cleanup() {
        إذا (بفر) {
            بفر.Destroy();
            بفر = تش_اشاره;
        }
    }

    ~webgpu_param_arena() { هذا->cleanup(); }
};

بنية webgpu_encoded_op {
    uint32_t عدد_kernels = 0;
#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_نومونه;
#د_که_پای
};

بنية webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_ډله_ثبت> bind_ډله_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

بنية webgpu_capabilities {
    wgpu::Limits limits;
    منطقي         supports_subgroups       = خطأ;
    منطقي         supports_subgroup_ماتریکس = خطأ;
    منطقي         supports_dot_حاصل_ضرب     = خطأ;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_اندازه     = 0;
    uint32_t min_subgroup_اندازه = 0;
    uint32_t max_subgroup_اندازه = 0;
    size_t   memset_بایټونه_فی_thread;
};

// Stores global webgpu members
بنية webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::کتار_2    queue;
    uint32_t       command_submit_batch_اندازه = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_اندازه;
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
#که_تعریف_شوی GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, مزدوج> cpu_وخت_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, مزدوج> cpu_detail_ms;
#د_که_پای

#که_تعریف_شوی GGML_WEBGPU_DEBUG
    wgpu::بفر_2 debug_host_buf;
    wgpu::بفر_2 debug_dev_buf;
#د_که_پای

    ~webgpu_global_context_struct() {
        إذا (هذا->get_tensor_staging_buf) {
            هذا->get_tensor_staging_buf.Destroy();
            هذا->get_tensor_staging_buf = تش_اشاره;
        }
        إذا (هذا->memset_params_buf) {
            هذا->memset_params_buf.Destroy();
            هذا->memset_params_buf = تش_اشاره;
        }
#که_تعریف_شوی GGML_WEBGPU_DEBUG
        إذا (هذا->debug_host_buf) {
            هذا->debug_host_buf.Destroy();
            هذا->debug_host_buf = تش_اشاره;
        }
        إذا (هذا->debug_dev_buf) {
            هذا->debug_dev_buf.Destroy();
            هذا->debug_dev_buf = تش_اشاره;
        }
#د_که_پای
    }
};

تعريفنوع std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
بنية webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::بفر_2             سېټ_لیکې_dev_error_buf;
    wgpu::بفر_2             سېټ_لیکې_host_error_buf;
    wgpu::CommandEncoder     فعال_command_encoder;
    wgpu::ComputePassEncoder فعال_compute_pass;
    منطقي                     batch_compute_passes = ریښتیا;

    size_t memset_بایټونه_فی_thread;

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, مزدوج> shader_gpu_وخت_ms;
    wgpu::بفر_2                            profile_timestamp_dev_buf;
    wgpu::بفر_2                            profile_timestamp_host_buf;
    wgpu::Query_سېټ                          profile_timestamp_query_سېټ;
    uint32_t                                profile_timestamp_query_شمېر = 0;
#د_که_پای

    ~webgpu_context_struct() {
#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
        إذا (هذا->profile_timestamp_host_buf) {
            هذا->profile_timestamp_host_buf.Destroy();
            هذا->profile_timestamp_host_buf = تش_اشاره;
        }
        إذا (هذا->profile_timestamp_dev_buf) {
            هذا->profile_timestamp_dev_buf.Destroy();
            هذا->profile_timestamp_dev_buf = تش_اشاره;
        }
        إذا (هذا->profile_timestamp_query_سېټ) {
            هذا->profile_timestamp_query_سېټ.Destroy();
            هذا->profile_timestamp_query_سېټ = تش_اشاره;
        }
#د_که_پای
        إذا (هذا->سېټ_لیکې_host_error_buf) {
            هذا->سېټ_لیکې_host_error_buf.Destroy();
            هذا->سېټ_لیکې_host_error_buf = تش_اشاره;
        }
        إذا (هذا->سېټ_لیکې_dev_error_buf) {
            هذا->سېټ_لیکې_dev_error_buf.Destroy();
            هذا->سېټ_لیکې_dev_error_buf = تش_اشاره;
        }
    }
};

تعريفنوع std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
بنية ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_شمېر;
    ثابت حرف *          name;
};

// Per-device struct for the global logical device interface
بنية ggml_backend_webgpu_device_context {
    webgpu_global_context webgpu_global_ctx;
    std::string           device_name;
    std::string           device_desc;
};

// Per-thread data required to actually run WebGPU operations in a backend instance
بنية ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;
    std::string    name;
};

// Per-thread data related to buffers
بنية ggml_backend_webgpu_بفر_context {
    wgpu::بفر_2          بفر;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_بفر_context(wgpu::بفر_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        بفر(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

ساكن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ثابت حرف *                             shader_کوډ,
                                                   ثابت حرف *                             label,
                                                   ثابت std::vector<wgpu::Constant_ثبت> & constants = {}) {
    wgpu::Shader_سرچینه_WGSL shader_سرچینه;
    shader_سرچینه.code = shader_کوډ;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.بل_دننه_Chain = &shader_سرچینه;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.ثبت_ټکی_2 = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = تش_اشاره;  // nullptr means auto layout
    إذا (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.معلومات();
        pipeline_desc.compute.constant_شمېر = constants.size();
    }
    أرجع { device.CreateComputePipeline(&pipeline_desc), label };
}

ساكن فراغ ggml_webgpu_create_بفر(wgpu::Device &    device,
                                      wgpu::بفر_2 &    بفر,
                                      size_t            size,
                                      wgpu::بفر_Usage usage,
                                      ثابت حرف *      label) {
    wgpu::بفر_Descriptor بفر_desc;
    بفر_desc.size             = size;
    بفر_desc.usage            = usage;
    بفر_desc.label            = label;
    بفر_desc.mappedAtCreation = خطأ;

    // TODO: error handling
    بفر = device.Create_بفر(&بفر_desc);
}

ساكن wgpu::بفر_2 ggml_webgpu_tensor_buf(ثابت ggml_tensor * tensor) {
    ggml_backend_webgpu_بفر_context * ctx = (ggml_backend_webgpu_بفر_context *) tensor->بفر->context;
    أرجع ctx->بفر;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
ساكن size_t ggml_webgpu_tensor_align_دننه_واټن(ثابت ggml_tensor * t, size_t alignment) {
    ثابت size_t دننه_واټن    = ggml_webgpu_tensor_دننه_واټن(t);
    ثابت size_t type_اندازه = ggml_type_اندازه(t->type);
    size_t       aligned   = دننه_واټن & ~(alignment - 1);
    طالما ((دننه_واټن - aligned) % type_اندازه != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    أرجع aligned;
}

ساكن size_t ggml_webgpu_tensor_misalignment(ثابت ggml_tensor * t, size_t alignment) {
    أرجع ggml_webgpu_tensor_دننه_واټن(t) - ggml_webgpu_tensor_align_دننه_واټن(t, alignment);
}

ساكن size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ثابت ggml_tensor * t) {
    أرجع ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment);
}

ساكن size_t ggml_webgpu_tensor_align_دننه_واټن(webgpu_context & ctx, ثابت ggml_tensor * t) {
    أرجع ggml_webgpu_tensor_align_دننه_واټن(t, ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment);
}

ساكن size_t ggml_webgpu_tensor_binding_اندازه(ثابت ggml_tensor * t, size_t alignment) {
    أرجع ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

ساكن size_t ggml_webgpu_tensor_binding_اندازه(webgpu_context & ctx, ثابت ggml_tensor * t) {
    أرجع ggml_webgpu_tensor_binding_اندازه(t, ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment);
}

ساكن منطقي ggml_webgpu_tensor_binding_overlap(ثابت webgpu_global_context & global_ctx,
                                               ثابت ggml_tensor *           a,
                                               ثابت ggml_tensor *           b) {
    إذا (a->بفر != b->بفر) {
        أرجع خطأ;
    }

    ثابت size_t alignment = global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment;
    ثابت size_t a_دننه_واټن  = ggml_webgpu_tensor_align_دننه_واټن(a, alignment);
    ثابت size_t b_دننه_واټن  = ggml_webgpu_tensor_align_دننه_واټن(b, alignment);
    أرجع a_دننه_واټن < b_دننه_واټن + ggml_webgpu_tensor_binding_اندازه(b, alignment) &&
           b_دننه_واټن < a_دننه_واټن + ggml_webgpu_tensor_binding_اندازه(a, alignment);
}

ساكن منطقي ggml_webgpu_tensor_binding_overlap_range(ثابت webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_بفر_t         بفر,
                                                     size_t                        دننه_واټن,
                                                     size_t                        size) {
    إذا (tensor->بفر != بفر) {
        أرجع خطأ;
    }

    ثابت size_t alignment     = global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment;
    ثابت size_t tensor_دننه_واټن = ggml_webgpu_tensor_align_دننه_واټن(tensor, alignment);
    أرجع tensor_دننه_واټن < دننه_واټن + size && دننه_واټن < tensor_دننه_واټن + ggml_webgpu_tensor_binding_اندازه(tensor, alignment);
}

بنية ggml_webgpu_merged_binding_range {
    size_t دننه_واټن;
    size_t size;
};

ساكن ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_دننه_واټن = SIZE_MAX;
    size_t merged_پای_ټکی    = 0;

    لكل (ggml_tensor * tensor : tensors) {
        ثابت size_t bind_دننه_واټن = ggml_webgpu_tensor_align_دننه_واټن(ctx, tensor);
        ثابت size_t bind_پای_ټکی    = bind_دننه_واټن + ggml_webgpu_tensor_binding_اندازه(ctx, tensor);

        merged_دننه_واټن = std::min(merged_دننه_واټن, bind_دننه_واټن);
        merged_پای_ټکی    = std::max(merged_پای_ټکی, bind_پای_ټکی);
    }

    أرجع { merged_دننه_واټن, merged_پای_ټکی - merged_دننه_واټن };
}

ساكن uint32_t ggml_webgpu_tensor_merged_element_دننه_واټن(ثابت ggml_tensor *                      tensor,
                                                         ثابت ggml_webgpu_merged_binding_range & merged_range) {
    أرجع (uint32_t) ((ggml_webgpu_tensor_دننه_واټن(tensor) - merged_range.دننه_واټن) / ggml_type_اندازه(tensor->type));
}

ساكن wgpu::Bind_ډله_ثبت ggml_webgpu_make_bind_ډله_ثبت(uint32_t     binding,
                                                              wgpu::بفر_2 بفر,
                                                              uint64_t     دننه_واټن,
                                                              uint64_t     size) {
    wgpu::Bind_ډله_ثبت ثبت = {};
    ثبت.binding              = binding;
    ثبت.بفر               = std::move(بفر);
    ثبت.دننه_واټن               = دننه_واټن;
    ثبت.size                 = size;
    أرجع ثبت;
}

ساكن wgpu::Bind_ډله_ثبت ggml_webgpu_make_tensor_bind_ډله_ثبت(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    أرجع ggml_webgpu_make_bind_ډله_ثبت(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_دننه_واټن(ctx, tensor),
                                             ggml_webgpu_tensor_binding_اندازه(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

کينډۍ <ډول_نوم T>
ساكن فراغ ggml_backend_webgpu_check_wait_حالت(wgpu::Wait_حالت wait_حالت,
                                                  T                callback_حالت,
                                                  T                success_حالت,
                                                  ثابت حرف *     wait_name,
                                                  ثابت حرف *     failure_name,
                                                  ثابت حرف *     callback_پیغام) {
    إذا (wait_حالت == wgpu::Wait_حالت::Timed_بهر) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    إذا (wait_حالت == wgpu::Wait_حالت::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    إذا (callback_حالت != success_حالت) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<صحيح>(callback_حالت),
                   callback_پیغام);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
ساكن uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    أرجع UINT32_MAX;
}

ساكن uint32_t ggml_backend_webgpu_get_command_submit_batch_اندازه() {
    أرجع WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_اندازه;
}

ساكن فراغ ggml_backend_webgpu_wait_کتار(webgpu_global_context & ctx) {
    wgpu::کتار_Work_بشپړ_حالت callback_حالت = wgpu::کتار_Work_بشپړ_حالت::Error;
    std::string               callback_پیغام;

    ثابت wgpu::Wait_حالت wait_حالت = ctx->instance.WaitAny(
        ctx->queue.On_Submitted_Work_بشپړ(
            wgpu::Callback_ډېرپېښ::AllowSpontaneous,
            [&callback_حالت, &callback_پیغام](wgpu::کتار_Work_بشپړ_حالت حالت, wgpu::StringView پیغام) {
                callback_حالت  = حالت;
                callback_پیغام = std::string(پیغام);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_حالت(wait_حالت, callback_حالت, wgpu::کتار_Work_بشپړ_حالت::Success,
                                          "Queue wait", "Queue work", callback_پیغام.c_str());
}

ساكن فراغ ggml_backend_webgpu_map_بفر(webgpu_global_context & ctx,
                                           wgpu::بفر_2 &          بفر,
                                           wgpu::Map_ډېرپېښ           mode,
                                           size_t                  دننه_واټن,
                                           size_t                  size) {
    wgpu::Map_Async_حالت callback_حالت = wgpu::Map_Async_حالت::Error;
    std::string          callback_پیغام;

    ثابت wgpu::Wait_حالت wait_حالت = ctx->instance.WaitAny(
        بفر.MapAsync(mode, دننه_واټن, size, wgpu::Callback_ډېرپېښ::AllowSpontaneous,
                        [&callback_حالت, &callback_پیغام](wgpu::Map_Async_حالت حالت, wgpu::StringView پیغام) {
                            callback_حالت  = حالت;
                            callback_پیغام = std::string(پیغام);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_حالت(wait_حالت, callback_حالت, wgpu::Map_Async_حالت::Success,
                                          "Buffer map wait", "Buffer map", callback_پیغام.c_str());
}

ساكن فراغ ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ثابت wgpu::Command_بفر commands,
                                                uint32_t &                عدد_inflight_batches) {
    إذا (عدد_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_کتار(ctx->global_ctx);
        عدد_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    عدد_inflight_batches++;
}

#که_تعریف_شوی GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
ساكن فراغ ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_بفر_منزل_بفر(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_اندازه());
    wgpu::Command_بفر commands = encoder.پای();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_بفر(ctx, ctx->debug_host_buf, wgpu::Map_ډېرپېښ::Read, 0, ctx->debug_host_buf.Get_اندازه());
    ثابت عائم * debug_معلومات = (ثابت عائم *) ctx->debug_host_buf.GetConstMappedRange();
    std::وتنه << "debug[0]: " << debug_معلومات[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#د_که_پای

ساكن webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ثابت std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            پایله = {};
    std::vector<wgpu::Bind_ډله> bind_groups;
    std::vector<size_t>          param_offsets;
    پایله.عدد_kernels = dispatches.size();

    لكل (size_t i = 0; i < dispatches.size(); i++) {
        ثابت webgpu_dispatch_desc & dispatch     = dispatches[i];
        ثابت size_t                 param_اندازه   = dispatch.params.size() * حجم(uint32_t);
        ثابت size_t                 param_دننه_واټن = ctx->param_arena.alloc_خونه(param_اندازه);

        std::vector<wgpu::Bind_ډله_ثبت> entries            = dispatch.bind_ډله_entries;
        uint32_t                          params_binding_عدد = entries.size();
        entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(params_binding_عدد, ctx->param_arena.بفر, param_دننه_واټن,
                                                            ctx->param_arena.خونه_اندازه));

        wgpu::Bind_ډله_Descriptor bind_ډله_desc;
        bind_ډله_desc.layout     = dispatch.pipeline.pipeline.Get_Bind_ډله_Layout(0);
        bind_ډله_desc.ثبت_شمېر = entries.size();
        bind_ډله_desc.entries    = entries.معلومات();
        bind_ډله_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.Create_Bind_ډله(&bind_ډله_desc));
        param_offsets.push_back(param_دننه_واټن);
    }

    لكل (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_بفر(ctx->param_arena.بفر, param_offsets[i], dispatches[i].params.معلومات(),
                                           dispatches[i].params.size() * حجم(uint32_t));
    }

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    لكل (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_شمېر + 2 <= WEBGPU_MAX_PROFILE_QUERY_شمېر);
        ثابت uint32_t query_begin = ctx->profile_timestamp_query_شمېر++;
        ثابت uint32_t query_پای_ټکی   = ctx->profile_timestamp_query_شمېر++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.query_سېټ                    = ctx->profile_timestamp_query_سېټ;
        ts_writes.beginning_د_Pass_Write_شاخص   = query_begin;
        ts_writes.پای_ټکی_د_Pass_Write_شاخص         = query_پای_ټکی;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->فعال_command_encoder.BeginComputePass(&pass_desc);

        pass.سېټ_Pipeline(dispatches[i].pipeline.pipeline);
        pass.سېټ_Bind_ډله(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.لومړی, dispatches[i].workgroups.second, 1);
        pass.پای_ټکی_2();
        پایله.pipeline_نومونه.push_back(dispatches[i].pipeline.name);
    }
#وإلا
    لكل (size_t i = 0; i < dispatches.size(); i++) {
        إذا (ctx->batch_compute_passes) {
            ctx->فعال_compute_pass.سېټ_Pipeline(dispatches[i].pipeline.pipeline);
            ctx->فعال_compute_pass.سېټ_Bind_ډله(0, bind_groups[i]);
            ctx->فعال_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.لومړی, dispatches[i].workgroups.second,
                                                        1);
        } وإلا {
            wgpu::ComputePassEncoder pass = ctx->فعال_command_encoder.BeginComputePass();
            pass.سېټ_Pipeline(dispatches[i].pipeline.pipeline);
            pass.سېټ_Bind_ډله(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.لومړی, dispatches[i].workgroups.second, 1);
            pass.پای_ټکی_2();
        }
    }
#د_که_پای

    أرجع پایله;
}

ساكن webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::Bind_ډله_ثبت> bind_ډله_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    أرجع ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_ډله_entries), { wg_x, wg_y } },
    });
}

ساكن فراغ ggml_backend_webgpu_بفر_memset(webgpu_global_context & ctx,
                                              wgpu::بفر_2 &          buf,
                                              uint32_t                value,
                                              size_t                  دننه_واټن,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) دننه_واټن, (uint32_t) size, value };
    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_bind_ډله_ثبت(0, buf, 0, buf.Get_اندازه()) };
    size_t                            بایټونه_فی_wg =
        ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup * ctx->capabilities.memset_بایټونه_فی_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, بایټونه_فی_wg);

    ctx->queue.Write_بفر(ctx->memset_params_buf, 0, params.معلومات(), params.size() * حجم(uint32_t));

    wgpu::Bind_ډله_ثبت params_ثبت = {};
    params_ثبت.binding              = 1;
    params_ثبت.بفر               = ctx->memset_params_buf;
    params_ثبت.دننه_واټن               = 0;
    params_ثبت.size                 = WEBGPU_PARAMS_BUF_اندازه_بایټونه;
    entries.push_back(params_ثبت);

    wgpu::Bind_ډله_Descriptor bind_ډله_desc;
    bind_ډله_desc.layout     = ctx->memset_pipeline.pipeline.Get_Bind_ډله_Layout(0);
    bind_ډله_desc.ثبت_شمېر = entries.size();
    bind_ډله_desc.entries    = entries.معلومات();
    bind_ډله_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::Bind_ډله bind_ډله = ctx->device.Create_Bind_ډله(&bind_ډله_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.سېټ_Pipeline(ctx->memset_pipeline.pipeline);
    pass.سېټ_Bind_ډله(0, bind_ډله);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.پای_ټکی_2();

    wgpu::Command_بفر              command  = encoder.پای();
    std::vector<wgpu::Command_بفر> commands = { command };
    ctx->queue.Submit(commands.size(), commands.معلومات());
}

/** End WebGPU Actions */

/** GGML Backend Interface */

ساكن ثابت حرف * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    أرجع ctx->name.c_str();
}

ساكن فراغ ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *) backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

#که_تعریف_شوی GGML_WEBGPU_CPU_PROFILE
    std::وتنه << "\n[ggml_webgpu cpu profiling summary]\n";
    مزدوج ټول_cpu = 0.0;
    لكل (ثابت خپلکاری & kv : ctx->webgpu_ctx->global_ctx->cpu_وخت_ms) {
        ټول_cpu += kv.second;
    }
    std::وتنه << "ggml_webgpu: total cpu time: " << ټول_cpu << " ms\n";
    std::وتنه << "ggml_webgpu: cpu breakdown:\n";
    لكل (ثابت خپلکاری & kv : ctx->webgpu_ctx->global_ctx->cpu_وخت_ms) {
        مزدوج pct = (ټول_cpu > 0.0) ? (kv.second / ټول_cpu * 100.0) : 0.0;
        std::وتنه << "ggml_webgpu:  " << kv.لومړی << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    إذا (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::وتنه << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    لكل (ثابت خپلکاری & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        مزدوج pct = (ټول_cpu > 0.0) ? (kv.second / ټول_cpu * 100.0) : 0.0;
        std::وتنه << "ggml_webgpu:  " << kv.لومړی << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#د_که_پای

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    std::وتنه << "\n[ggml_webgpu gpu profiling summary]\n";
    مزدوج ټول_gpu = 0.0;
    لكل (ثابت خپلکاری & kv : ctx->webgpu_ctx->shader_gpu_وخت_ms) {
        ټول_gpu += kv.second;
    }
    std::وتنه << "ggml_webgpu: total gpu time (all shaders): " << ټول_gpu << " ms\n";
    std::وتنه << "\nggml_webgpu: gpu breakdown:\n";
    لكل (ثابت خپلکاری & kv : ctx->webgpu_ctx->shader_gpu_وخت_ms) {
        مزدوج pct = (ټول_gpu > 0.0) ? (kv.second / ټول_gpu * 100.0) : 0.0;
        std::وتنه << "ggml_webgpu:  " << kv.لومړی << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#د_که_پای

#إذا defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::وتنه << "ggml_webgpu: gpu/cpu ratio: " << (ټول_cpu > 0.0 ? ټول_gpu / ټول_cpu : 0.0) << "\n";
#د_که_پای

    ړنګول ctx;
    ړنګول backend;
}

ساكن webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne, (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[0] / ggml_type_اندازه(src->type)), (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)), (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),
        (uint32_t) (dst->nb[0] / ggml_type_اندازه(dst->type)), (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)), (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
        // Logical shapes
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1], (uint32_t) dst->ne[2]
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_سېټ(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_سېټ_pipeline(shader_lib_ctx);

    خپلکاری *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ثابت منطقي inplace   = decisions->inplace;

    ثابت uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
    ثابت uint32_t dst_type_اندازه = (uint32_t) ggml_type_اندازه(dst->type);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (((ثابت int32_t *) dst->op_params)[3] / dst_type_اندازه),

        (uint32_t) (src1->nb[0] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),

        1u,
        (uint32_t) (((ثابت int32_t *) dst->op_params)[0] / dst_type_اندازه),
        (uint32_t) (((ثابت int32_t *) dst->op_params)[1] / dst_type_اندازه),
        (uint32_t) (((ثابت int32_t *) dst->op_params)[2] / dst_type_اندازه),

        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries;
    uint32_t                          binding_شاخص = 0;
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0));
        binding_شاخص++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_شاخص, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_شاخص + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        ne,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        // Strides (in elements)
        (uint32_t) (src->nb[0] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),
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

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.wg_mem_حد_بایټونه = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_اندازه;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst),
    };

    ثابت uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_اندازه);
    ثابت uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
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
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),

        (uint32_t) (src0->nb[0] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),

        (uint32_t) (src1->nb[0] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),

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

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Same param/binding layout as conv_2d; the shader differs
ساكن webgpu_encoded_op ggml_webgpu_conv_2d_dw(webgpu_context & ctx,
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
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),

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

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ثابت منطقي      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    خپلکاری *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    ثابت int32_t s0    = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t s1    = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t p0    = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t p1    = ggml_get_op_params_i32(dst, 3);
    ثابت int32_t d0    = ggml_get_op_params_i32(dst, 4);
    ثابت int32_t d1    = ggml_get_op_params_i32(dst, 5);
    ثابت منطقي    is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    ثابت uint32_t KW = src0->ne[0];
    ثابت uint32_t KH = is_2D ? src0->ne[1] : 1;
    ثابت uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];

    ثابت uint32_t IW = src1->ne[0];
    ثابت uint32_t IH = is_2D ? src1->ne[1] : 1;
    ثابت uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];

    ثابت uint32_t OW = dst->ne[1];
    ثابت uint32_t OH = is_2D ? dst->ne[2] : 1;

    ثابت uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_اندازه(src1->type));
    ثابت uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)) : 0;
    ثابت uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)) :
                                 (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type));
    ثابت uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)) :
                                 (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type));

    ثابت uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_اندازه(dst->type));
    ثابت uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type));
    ثابت uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)) : 0;
    ثابت uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)) :
                                 (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type));

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),

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

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    خپلکاری *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->نښې_فی_wg);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),

        (uint32_t) (dst->nb[0] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),

        (uint32_t) src1->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        token_tiles,
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst),
    };

    ثابت uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->بلاک_اندازه_3);
    ثابت uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_ssm_لټون(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    uint8_t xbc_overlap = 0;
    إذا (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src4)) {
        xbc_overlap |= 0b110;  // x/B
    }
    إذا (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src1, src5)) {
        xbc_overlap |= 0b101;  // x/C
    }
    إذا (ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src4, src5)) {
        xbc_overlap |= 0b011;  // B/C
    }

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_لټون_pipeline(shader_lib_ctx, xbc_overlap);
    خپلکاری *          decisions = static_cast<ggml_webgpu_ssm_لټون_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t دننه_واټن_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    uint32_t دننه_واټن_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type));
    uint32_t دننه_واټن_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_اندازه(src3->type));
    uint32_t دننه_واټن_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_اندازه(src4->type));
    uint32_t دننه_واټن_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_اندازه(src5->type));
    uint32_t دننه_واټن_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_اندازه(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    إذا (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        دننه_واټن_x         = ggml_webgpu_tensor_merged_element_دننه_واټن(src1, xbc_merged_range);
        دننه_واټن_B         = ggml_webgpu_tensor_merged_element_دننه_واټن(src4, xbc_merged_range);
    } وإلا إذا (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        دننه_واټن_B         = ggml_webgpu_tensor_merged_element_دننه_واټن(src4, xbc_merged_range);
        دننه_واټن_C         = ggml_webgpu_tensor_merged_element_دننه_واټن(src5, xbc_merged_range);
    } وإلا إذا (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        دننه_واټن_x         = ggml_webgpu_tensor_merged_element_دننه_واټن(src1, xbc_merged_range);
        دننه_واټن_B         = ggml_webgpu_tensor_merged_element_دننه_واټن(src4, xbc_merged_range);
        دننه_واټن_C         = ggml_webgpu_tensor_merged_element_دننه_واټن(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_دننه_واټن = 0;
    size_t xbc_bind_اندازه   = 0;
    إذا (xbc_overlap > 0) {
        xbc_bind_دننه_واټن = xbc_merged_range.دننه_واټن;
        xbc_bind_اندازه   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        دننه_واټن_x,
        دننه_واټن_dt,
        دننه_واټن_A,
        دننه_واټن_B,
        دننه_واټن_C,
        دننه_واټن_ids,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),

        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),

        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),

        (uint32_t) (src2->nb[1] / ggml_type_اندازه(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_اندازه(src2->type)),

        (uint32_t) src3->ne[0],
        (uint32_t) (src3->nb[1] / ggml_type_اندازه(src3->type)),

        (uint32_t) (src4->nb[1] / ggml_type_اندازه(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_اندازه(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_اندازه(src4->type)),

        (uint32_t) (src5->nb[1] / ggml_type_اندازه(src5->type)),
        (uint32_t) (src5->nb[2] / ggml_type_اندازه(src5->type)),
        (uint32_t) (src5->nb[3] / ggml_type_اندازه(src5->type)),

        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src4->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) ggml_nelements(src1),
        (uint32_t) ggml_get_op_params_i32(dst, 0),
    };

    uint32_t binding_عدد = 0;

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, src0),
    };
    // xbc_merged binding
    إذا (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(binding_عدد++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_دننه_واټن, xbc_bind_اندازه));
    }
    // x
    إذا (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, src1));
    }
    // B
    إذا (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, src4));
    }
    // C
    إذا (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد++, dst));

    ثابت uint32_t ټول_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(ټول_wg, max_wg_فی_dim, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ثابت uint32_t s_v      = (uint32_t) src2->ne[0];
    ثابت uint32_t h        = (uint32_t) src2->ne[1];
    ثابت uint32_t n_نښې = (uint32_t) src2->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ثابت uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ثابت عائم    scale    = 1.0f / sqrtf((عائم) s_v);
    uint32_t       scale_u32;
    حافظه_نقل(&scale_u32, &scale, حجم(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_نښې,
        n_seqs,
        s_v * h * n_نښې * n_seqs,

        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),

        (uint32_t) (src2->nb[1] / ggml_type_اندازه(src2->type)),
        (uint32_t) (src2->nb[2] / ggml_type_اندازه(src2->type)),
        (uint32_t) (src2->nb[3] / ggml_type_اندازه(src2->type)),

        (uint32_t) (src4->nb[1] / ggml_type_اندازه(src4->type)),
        (uint32_t) (src4->nb[2] / ggml_type_اندازه(src4->type)),
        (uint32_t) (src4->nb[3] / ggml_type_اندازه(src4->type)),

        (uint32_t) src0->ne[1],
        (uint32_t) (src2->ne[3] / src0->ne[3]),
        K,
        scale_u32,
        dst_fuse ? (uint32_t) (dst_fuse->nb[2] / ggml_type_اندازه(dst_fuse->type)) : 0,
        dst_fuse ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst_fuse) / ggml_type_اندازه(dst_fuse->type)) : 0,
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0), ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, src2), ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 4, src4), ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 6, dst),
    };

    إذا (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 7, dst_fuse));
    }

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

ساكن std::optional<webgpu_encoded_op> ggml_webgpu_سېټ_لیکې(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    إذا (ggml_is_تشه(src) || ggml_is_تشه(idx)) {
        أرجع std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_سېټ_لیکې_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_سېټ_لیکې_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_اندازه(idx->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        // Convert byte-strides to element-strides
        (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)), (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)), (uint32_t) (idx->nb[0] / ggml_type_اندازه(idx->type)),
        (uint32_t) (idx->nb[1] / ggml_type_اندازه(idx->type)), (uint32_t) (idx->nb[2] / ggml_type_اندازه(idx->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)), (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
        // Shape of src
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        // Shape of idx
        (uint32_t) (idx->ne[1]), (uint32_t) (idx->ne[2])
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst));

    إذا (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(3, ctx->سېټ_لیکې_dev_error_buf, 0,
                                                            ctx->سېټ_لیکې_dev_error_buf.Get_اندازه()));
    }

    uint32_t تارونه;
    إذا (ggml_is_quantized(dst->type)) {
        ثابت uint32_t بلاکونه_فی_لیکه = src->ne[0] / ggml_blck_اندازه(dst->type);
        تارونه =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_بلاکونه ? (بلاکونه_فی_لیکه / 2) : بلاکونه_فی_لیکه);
    } وإلا إذا (decisions->vec4) {
        تارونه = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } وإلا {
        تارونه = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(تارونه, decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
ساكن std::vector<wgpu::Constant_ثبت> ggml_webgpu_wg_اندازه_ثبت(uint32_t wg_اندازه) {
    std::vector<wgpu::Constant_ثبت> constants(1);
    constants[0].کلی   = "wg_size";
    constants[0].value = wg_اندازه;
    أرجع constants;
}

ساكن webgpu_encoded_op ggml_webgpu_get_لیکې(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ثابت منطقي float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = تش_اشاره;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت uint32_t دننه_واټن_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type));
    ثابت uint32_t دننه_واټن_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type));
    ثابت uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type));

    ثابت منطقي vec4_aligned = دننه_واټن_src % 4 == 0 && دننه_واټن_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_لیکې_pipeline(shader_lib_ctx, vec4_aligned);
    خپلکاری *          decisions = static_cast<ggml_webgpu_get_لیکې_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { دننه_واټن_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_اندازه(idx->type)),
                                     دننه_واټن_dst,
                                     stride_src1,
                                     stride_src2,
                                     stride_src3,
                                     (uint32_t) (idx->nb[0] / ggml_type_اندازه(idx->type)),
                                     (uint32_t) (idx->nb[1] / ggml_type_اندازه(idx->type)),
                                     (uint32_t) (idx->nb[2] / ggml_type_اندازه(idx->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],
                                     (uint32_t) (idx->ne[1]),
                                     (uint32_t) (idx->ne[2]) };

    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst) };

    uint32_t بلاکونه_فی_لیکه = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t ټول_لیکې     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t ټول_تارونه  = float_parallel ? بلاکونه_فی_لیکه * ټول_لیکې : ټول_لیکې;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ټول_تارونه, decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن فراغ ggml_webgpu_quantize_q8_dispatch(webgpu_context &                    ctx,
                                             ggml_tensor *                       src0,
                                             ggml_tensor *                       src1,
                                             ggml_tensor *                       dst,
                                             std::vector<webgpu_dispatch_desc> & dispatches) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0               = src0;
    shader_lib_ctx.src1               = src1;
    shader_lib_ctx.dst                = dst;
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ثابت size_t dst_دننه_واټن           = ggml_webgpu_tensor_دننه_واټن(dst);
    ثابت size_t q8_src1_align_دننه_واټن = ROUNDUP_POW2(
        dst_دننه_واټن + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment);
    ثابت size_t q8_src1_binding_اندازه = ROUNDUP_POW2(
        src1->ne[3] * src1->ne[2] * src1->ne[1] * (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32)),
        WEBGPU_STORAGE_BUF_BINDING_MULT);

    std::vector<uint32_t> q8_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ډله_ثبت> q8_entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src1),
        ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_دننه_واټن, q8_src1_binding_اندازه)
    };

    خپلکاری q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_اندازه     = q8_decisions->wg_اندازه;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ثابت uint32_t wg_فی_vec     = (src0->ne[0] / 4 + (q8_wg_اندازه - 1)) / q8_wg_اندازه;
    ثابت uint32_t q8_ټول_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_فی_vec;
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    compute_2d_workgroups(q8_ټول_wg, max_wg_فی_dim, q8_wg_x, q8_wg_y);

    dispatches.push_back({
        qq8_pipeline, std::move(q8_params), std::move(q8_entries), { q8_wg_x, q8_wg_y }
    });
}

ساكن webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                                             ggml_tensor *    src0,
                                             ggml_tensor *    src1,
                                             ggml_tensor *    dst) {
    // Determine if this is a mat-vec operation
    منطقي use_mat_vec = (dst->ne[1] <= 4);

    // use MMVQ path for mat-vec
    منطقي use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_حاصل_ضرب,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_اندازه              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_ماتریکس = ctx->global_ctx->capabilities.supports_subgroup_ماتریکس;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_اندازه        = ctx->global_ctx->capabilities.min_subgroup_اندازه;
    shader_lib_ctx.max_subgroup_اندازه        = ctx->global_ctx->capabilities.max_subgroup_اندازه;
    shader_lib_ctx.supports_dot_حاصل_ضرب     = ctx->global_ctx->capabilities.supports_dot_حاصل_ضرب;
    shader_lib_ctx.vendor                   = ctx->global_ctx->vendor;

    // Get or create pipeline
    webgpu_pipeline                   pipeline;
    std::vector<webgpu_dispatch_desc> dispatches;
    ثابت منطقي src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;

    إذا (use_mat_vec) {
        إذا (use_mmvq) {
            ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
        }
        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
    } وإلا {
        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
    }

    uint32_t دننه_واټن_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t دننه_واټن_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    size_t   merged_دننه_واټن = 0;
    size_t   merged_اندازه   = 0;
    إذا (src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_دننه_واټن = merged_range.دننه_واټن;
        merged_اندازه   = merged_range.size;
        دننه_واټن_src0   = ggml_webgpu_tensor_merged_element_دننه_واټن(src0, merged_range);
        دننه_واټن_src1   = ggml_webgpu_tensor_merged_element_دننه_واټن(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { دننه_واټن_src0,
                                     دننه_واټن_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) src0->ne[0],
                                     (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),
                                     (uint32_t) src0->ne[2],
                                     (uint32_t) src0->ne[3],
                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };

    // Build bind group entries
    std::vector<wgpu::Bind_ډله_ثبت> entries = {};
    إذا (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0));
        خپلکاری & mmvq_qq8_ثبت = dispatches[0].bind_ډله_entries[1];
        entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_ثبت.دننه_واټن,
                                                            mmvq_qq8_ثبت.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst));
    } وإلا إذا (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src0), merged_دننه_واټن, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;

    إذا (use_mat_vec) {
        خپلکاری * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_فی_wg);
        uint32_t ټول_wg      = output_groups * batches;
        compute_2d_workgroups(ټول_wg, max_wg_فی_dim, wg_x, wg_y);
    } وإلا {
        خپلکاری * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        إذا (decisions->use_subgroup_ماتریکس) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_ماتریکس_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_ماتریکس_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } وإلا {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_اندازه_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_اندازه_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t ټول_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(ټول_wg, max_wg_فی_dim, wg_x, wg_y);
    }

    dispatches.push_back({
        pipeline, std::move(params), std::move(entries), { wg_x, wg_y }
    });

    أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساكن webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(webgpu_context & ctx,
                                                    ggml_tensor *    src0,
                                                    ggml_tensor *    src1,
                                                    ggml_tensor *    src2,
                                                    ggml_tensor *    dst) {
    ثابت uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ثابت uint32_t param_n_expert_کارول_شوی = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_کارول_شوی,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_دننه_واټن(ctx, src0),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src0)),
        ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_دننه_واټن(ctx, src1),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src1)),
        ggml_webgpu_make_bind_ډله_ثبت(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_دننه_واټن(ctx, src2),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src2)),
        ggml_webgpu_make_bind_ډله_ثبت(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_دننه_واټن(ctx, dst),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    خپلکاری * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_فی_wg);
    uint32_t       ټول_wg       = output_groups * param_n_expert_کارول_شوی;
    compute_2d_workgroups(ټول_wg, max_wg_فی_dim, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// upper bound on the number of slot tiles emitted by mul_mat_id_gather.wgsl
ساكن uint32_t ggml_webgpu_mul_mat_id_max_tiles(ثابت ggml_tensor * dst) {
    ثابت uint32_t n_expert = (uint32_t) dst->src[0]->ne[2];
    ثابت uint32_t n_slots  = (uint32_t) (dst->ne[1] * dst->ne[2]);  // n_expert_used * n_tokens

    أرجع CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_اندازه_N) + std::min(n_expert, n_slots);
}

// tile count followed by (expert, first slot, n slots) per tile
ساكن size_t ggml_webgpu_mul_mat_id_tiles_nbytes(ثابت ggml_tensor * dst) {
    أرجع (1 + 3 * (size_t) ggml_webgpu_mul_mat_id_max_tiles(dst)) * حجم(uint32_t);
}

ساكن webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    src2,
                                                ggml_tensor *    dst) {
    // we can use mat-vec fast path
    إذا (dst->ne[2] == 1) {
        أرجع ggml_webgpu_mul_mat_id_vec(ctx, src0, src1, src2, dst);
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ثابت uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ثابت uint32_t param_n_expert_کارول_شوی = (uint32_t) dst->ne[1];
    ثابت uint32_t param_n_نښې      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)),
        param_n_expert,
        param_n_expert_کارول_شوی,
        param_n_نښې,
        (uint32_t) (src2->nb[1] / ggml_type_اندازه(src2->type)),
    };

    ثابت size_t dst_دننه_واټن       = ggml_webgpu_tensor_دننه_واټن(dst);
    ثابت size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * حجم(uint32_t);
    ثابت size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ثابت size_t ids_cache_align_دننه_واټن = ROUNDUP_POW2(
        dst_دننه_واټن + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment);
    ثابت size_t ids_tiles_align_دننه_واټن =
        ROUNDUP_POW2(ids_cache_align_دننه_واټن + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment);

    ثابت size_t ids_cache_bind_اندازه = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ثابت size_t ids_tiles_bind_اندازه = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::Bind_ډله_ثبت> gather_entries = {
        ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_دننه_واټن(ctx, src2),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src2)),
        ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_دننه_واټن, ids_cache_bind_اندازه),
        ggml_webgpu_make_bind_ډله_ثبت(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_دننه_واټن, ids_tiles_bind_اندازه),
    };

    dispatches.push_back({
        gather_pipeline, std::move(gather_params), std::move(gather_entries), { 1, 1 }
    });

    // params for mul_mat_id.wgsl
    std::vector<uint32_t> main_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_کارول_شوی,
        param_n_نښې,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::Bind_ډله_ثبت> main_entries = {
        ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_دننه_واټن(ctx, src0),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src0)),
        ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_دننه_واټن(ctx, src1),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src1)),
        ggml_webgpu_make_bind_ډله_ثبت(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_دننه_واټن(ctx, dst),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, dst)),
        ggml_webgpu_make_bind_ډله_ثبت(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_دننه_واټن, ids_cache_bind_اندازه),
        ggml_webgpu_make_bind_ډله_ثبت(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_دننه_واټن, ids_tiles_bind_اندازه),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    خپلکاری * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_اندازه_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_اندازه_N);

    ثابت uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_اندازه_m;
    ثابت uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ثابت uint32_t ټول_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
}

بنية ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_ډله_ثبت> entries;
    size_t                            kv_bind_دننه_واټن = 0;
    size_t                            kv_bind_اندازه   = 0;
    منطقي                              has_ماسک       = خطأ;
    منطقي                              has_sinks      = خطأ;
    منطقي                              kv_overlap     = خطأ;
};

ساكن منطقي ggml_webgpu_flash_attn_use_vec_لار(ثابت webgpu_global_context & global_ctx,
                                                ثابت ggml_tensor *           Q,
                                                ثابت ggml_tensor *           K,
                                                ثابت ggml_tensor *           V) {
    ثابت size_t storage_دننه_واټن_alignment = global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment;

    ثابت منطقي k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_دننه_واټن_alignment);
    ثابت منطقي v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_دننه_واټن_alignment);

    ثابت uint32_t k_vec_سر_align =
        ggml_is_quantized(K->type) ? ggml_blck_اندازه(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی;
    ثابت uint32_t v_vec_سر_align =
        ggml_is_quantized(V->type) ? ggml_blck_اندازه(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی;
    ثابت منطقي kv_vec_سر_dims_aligned = Q->ne[0] % k_vec_سر_align == 0 && V->ne[0] % v_vec_سر_align == 0;

    أرجع global_ctx->capabilities.supports_subgroups && (Q->ne[1] < GGML_WEBGPU_FLASH_ATTN_VEC_MAX_SEQ_LEN) &&
           kv_vec_سر_dims_aligned && k_float_vec4_aligned && v_float_vec4_aligned;
}

ساكن ggml_webgpu_flash_attn_op ggml_webgpu_flash_attn_prepare(webgpu_context & ctx,
                                                                ggml_tensor *    Q,
                                                                ggml_tensor *    K,
                                                                ggml_tensor *    V,
                                                                ggml_tensor *    ماسک,
                                                                ggml_tensor *    sinks,
                                                                ggml_tensor *    dst) {
    عائم scale         = ggml_get_op_params_f32(dst, 0);
    عائم max_bias      = ggml_get_op_params_f32(dst, 1);
    عائم logit_softcap = ggml_get_op_params_f32(dst, 2);
    إذا (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    عائم n_سر_log2 = عائم(1u << (uint32_t) floor(log2(Q->ne[2])));
    عائم m0          = powf(2.0f, -(max_bias) / n_سر_log2);
    عائم m1          = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    ggml_webgpu_flash_attn_op op               = {};
    op.shader_lib_ctx.src0                     = Q;
    op.shader_lib_ctx.src1                     = K;
    op.shader_lib_ctx.src2                     = V;
    op.shader_lib_ctx.src3                     = ماسک;
    op.shader_lib_ctx.src4                     = sinks;
    op.shader_lib_ctx.dst                      = dst;
    op.shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    op.shader_lib_ctx.supports_subgroup_ماتریکس = ctx->global_ctx->capabilities.supports_subgroup_ماتریکس;
    op.shader_lib_ctx.max_wg_اندازه              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    op.shader_lib_ctx.wg_mem_حد_بایټونه       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_اندازه;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_اندازه        = ctx->global_ctx->capabilities.min_subgroup_اندازه;
    op.shader_lib_ctx.max_subgroup_اندازه        = ctx->global_ctx->capabilities.max_subgroup_اندازه;

    op.has_ماسک   = ماسک != تش_اشاره;
    op.has_sinks  = sinks != تش_اشاره;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t دننه_واټن_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_اندازه(K->type));
    uint32_t دننه_واټن_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_اندازه(V->type));
    إذا (op.kv_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_دننه_واټن                                   = merged_range.دننه_واټن;
        op.kv_bind_اندازه                                     = merged_range.size;
        دننه_واټن_k                                            = ggml_webgpu_tensor_merged_element_دننه_واټن(K, merged_range);
        دننه_واټن_v                                            = ggml_webgpu_tensor_merged_element_دننه_واټن(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_اندازه(Q->type)),
        دننه_واټن_k,
        دننه_واټن_v,
        op.has_ماسک ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ماسک) / ggml_type_اندازه(ماسک->type)) : 0,
        op.has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_اندازه(sinks->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) Q->ne[2],                              // number of heads
        (uint32_t) Q->ne[1],                              // sequence length (Q)
        (uint32_t) K->ne[1],                              // sequence length (K/V)
        (uint32_t) (Q->nb[1] / ggml_type_اندازه(Q->type)),  // stride (elements/blocks) of Q in dimension 1
        (uint32_t) (Q->nb[2] / ggml_type_اندازه(Q->type)),  // stride (elements/blocks) of Q in dimension 2
        (uint32_t) (Q->nb[3] / ggml_type_اندازه(Q->type)),  // stride (elements/blocks) of Q in dimension 3
        (uint32_t) (K->nb[1] / ggml_type_اندازه(K->type)),  // stride (elements/blocks) of K in dimension 1
        (uint32_t) (K->nb[2] / ggml_type_اندازه(K->type)),  // stride (elements/blocks) of K in dimension 2
        (uint32_t) (K->nb[3] / ggml_type_اندازه(K->type)),  // stride (elements/blocks) of K in dimension 3
        (uint32_t) (V->nb[1] / ggml_type_اندازه(V->type)),  // stride (elements/blocks) of V in dimension 1
        (uint32_t) (V->nb[2] / ggml_type_اندازه(V->type)),  // stride (elements/blocks) of V in dimension 2
        (uint32_t) (V->nb[3] / ggml_type_اندازه(V->type)),  // stride (elements/blocks) of V in dimension 3
        op.has_ماسک ? (uint32_t) (ماسک->nb[3] / ggml_type_اندازه(ماسک->type)) : 0,  // stride of mask dim 3
        (uint32_t) (Q->ne[2] / K->ne[2]),  // repeat factor for K/V in dim 2 (MHA/MQA/GQA)
        ggml_webgpu_u32_from_f32(scale),   // scale (possibly adjusted for logit softcap)
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(logit_softcap),
        ggml_webgpu_u32_from_f32(n_سر_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };
    op.entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, Q),
    };
    إذا (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(K), op.kv_bind_دننه_واټن, op.kv_bind_اندازه));
    } وإلا {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, V));
    }
    uint32_t binding_شاخص = op.kv_overlap ? 2u : 3u;
    إذا (op.has_ماسک) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_شاخص++, ماسک));
    }
    إذا (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_شاخص++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_شاخص++, dst));

    أرجع op;
}

ساكن uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ثابت uint64_t kv_پراختیا = (uint64_t) kv_tile;
    طالما ((2u * nwg * kv_پراختیا) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    أرجع std::min(nwg, vec_nwg_cap);
}

ساكن webgpu_encoded_op ggml_webgpu_flash_attn_مستقیم(webgpu_context & ctx, ثابت ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    خپلکاری *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_فی_سر = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_فی_سر * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
    أرجع ggml_backend_webgpu_build(ctx, pipeline, op.params, op.entries, wg_x);
}

ساكن webgpu_encoded_op ggml_webgpu_flash_attn_vec(webgpu_context &          ctx,
                                                    ggml_tensor *             Q,
                                                    ggml_tensor *             K,
                                                    ggml_tensor *             V,
                                                    ggml_tensor *             ماسک,
                                                    ggml_tensor *             sinks,
                                                    ggml_tensor *             dst,
                                                    ggml_webgpu_flash_attn_op op) {
    webgpu_pipeline pipeline  = ctx->shader_lib->get_flash_attn_vec_pipeline(op.shader_lib_ctx, op.kv_overlap);
    خپلکاری *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::بفر_2 blk_buf         = {};
    uint64_t     blk_اندازه_بایټونه  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_شمېر = 0;

    ثابت uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_اندازه;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ثابت uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ثابت منطقي     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_بنسټ  = 0;
    uint64_t     tmp_اندازه_بایټونه  = 0;
    wgpu::بفر_2 tmp_buf         = {};
    uint64_t     tmp_bind_دننه_واټن = 0;
    uint64_t     tmp_bind_اندازه   = 0;
    ثابت size_t align_بایټونه     = ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment;
    ثابت size_t dst_دننه_واټن      = ggml_webgpu_tensor_دننه_واټن(dst);
    size_t       scratch_دننه_واټن  = ROUNDUP_POW2(dst_دننه_واټن + ggml_nbytes(dst), align_بایټونه);

    إذا (use_vec_reduce) {
        ثابت uint64_t tmp_معلومات_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ثابت uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_بنسټ                 = tmp_معلومات_elems;
        tmp_اندازه_بایټونه =
            ROUNDUP_POW2((tmp_معلومات_elems + tmp_stats_elems) * حجم(عائم), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_بنسټ <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_دننه_واټن = scratch_دننه_واټن;
        tmp_bind_اندازه   = tmp_اندازه_بایټونه;
        scratch_دننه_واټن  = ROUNDUP_POW2(scratch_دننه_واټن + tmp_اندازه_بایټونه, align_بایټونه);
    } وإلا {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_اندازه_بایټونه  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_دننه_واټن = scratch_دننه_واټن;
        tmp_bind_اندازه   = tmp_اندازه_بایټونه;
        scratch_دننه_واټن  = ROUNDUP_POW2(scratch_دننه_واټن + tmp_اندازه_بایټونه, align_بایټونه);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::Bind_ډله_ثبت> blk_entries;
    إذا (op.has_ماسک) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ثابت uint32_t stride_mask3 = (uint32_t) (ماسک->nb[3] / ggml_type_اندازه(ماسک->type));
        blk_batch_شمېر             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ثابت uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_شمېر;
        blk_اندازه_بایټونه              = ROUNDUP_POW2(blk_elems * حجم(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
        ثابت ggml_webgpu_shader_lib_context blk_shader_ctx = op.shader_lib_ctx;
        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);

        blk_params = {
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, ماسک) / ggml_type_اندازه(ماسک->type)),  // offset_mask
            (uint32_t) Q->ne[1],                                                                   // seq_len_q
            (uint32_t) K->ne[1],                                                                   // seq_len_kv
            stride_mask3,                                                                          // stride_mask3
            blk_nblk0,                                                                             // nblk0
            blk_nblk1,                                                                             // nblk1
        };
        blk_entries = {
            ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(ماسک),
                                              ggml_webgpu_tensor_align_دننه_واټن(ctx, ماسک),
                                              ggml_webgpu_tensor_binding_اندازه(ctx, ماسک)),
            ggml_webgpu_make_bind_ډله_ثبت(1, blk_buf, scratch_دننه_واټن, blk_اندازه_بایټونه),
        };
        scratch_دننه_واټن = ROUNDUP_POW2(scratch_دننه_واټن + blk_اندازه_بایټونه, align_بایټونه);
    }

    std::vector<uint32_t> وېشل_params = op.params;
    إذا (op.has_ماسک) {
        وېشل_params.push_back(0u);                     // blk_base
        وېشل_params.push_back(blk_nblk0);              // blk_nblk0
        وېشل_params.push_back(blk_nblk1);              // blk_nblk1
    }
    وېشل_params.push_back(0u);                         // tmp_data_base
    وېشل_params.push_back((uint32_t) tmp_stats_بنسټ);  // tmp_stats_base
    وېشل_params.push_back(nwg);                        // nwg

    std::vector<wgpu::Bind_ډله_ثبت> وېشل_entries = {
        ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_دننه_واټن(ctx, Q),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, Q)),
    };
    إذا (op.kv_overlap) {
        وېشل_entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(K), op.kv_bind_دننه_واټن, op.kv_bind_اندازه));
    } وإلا {
        وېشل_entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_دننه_واټن(ctx, K),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, K)));
        وېشل_entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_دننه_واټن(ctx, V),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, V)));
    }
    uint32_t وېشل_binding_شاخص = op.kv_overlap ? 2u : 3u;
    إذا (op.has_ماسک) {
        وېشل_entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(وېشل_binding_شاخص++, ggml_webgpu_tensor_buf(ماسک),
                                                                  ggml_webgpu_tensor_align_دننه_واټن(ctx, ماسک),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, ماسک)));
    }
    إذا (op.has_sinks) {
        وېشل_entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(وېشل_binding_شاخص++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_دننه_واټن(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, sinks)));
    }
    إذا (op.has_ماسک) {
        وېشل_entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(وېشل_binding_شاخص++, blk_buf, blk_entries[1].دننه_واټن, blk_اندازه_بایټونه));
    }
    وېشل_entries.push_back(
        ggml_webgpu_make_bind_ډله_ثبت(وېشل_binding_شاخص++, tmp_buf, tmp_bind_دننه_واټن, tmp_bind_اندازه));
    وېشل_entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(وېشل_binding_شاخص++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_دننه_واټن(ctx, dst),
                                                              ggml_webgpu_tensor_binding_اندازه(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::Bind_ډله_ثبت> reduce_entries;
    إذا (use_vec_reduce) {
        ثابت uint32_t reduce_sg_اندازه = ctx->global_ctx->capabilities.max_subgroup_اندازه;
        ثابت uint32_t reduce_wg_اندازه = std::max(
            reduce_sg_اندازه,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_اندازه,
                                          ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup));
        ggml_webgpu_shader_lib_context reduce_shader_ctx = op.shader_lib_ctx;
        reduce_shader_ctx.max_wg_اندازه                    = reduce_wg_اندازه;
        reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);

        reduce_params = {
            (uint32_t) nrows,                                                                    // nrows
            (uint32_t) Q->ne[1],                                                                 // seq_len_q
            (uint32_t) Q->ne[2],                                                                 // n_heads
            (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),  // offset_dst
            nwg,                                                                                 // nwg
            0u,                                                                                  // tmp_data_base
            (uint32_t) tmp_stats_بنسټ,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_ډله_ثبت(0, tmp_buf, tmp_bind_دننه_واټن, tmp_اندازه_بایټونه),
            ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_دننه_واټن(ctx, dst),
                                              ggml_webgpu_tensor_binding_اندازه(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ثابت uint64_t وېشل_wg_ټول = (uint64_t) wg_x * nwg;
    GGML_ASSERT(وېشل_wg_ټول <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    إذا (op.has_ماسک) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_شمېر }
        });
    }
    dispatches.push_back({
        pipeline, std::move(وېشل_params), std::move(وېشل_entries), { (uint32_t) وېشل_wg_ټول, 1u }
    });
    إذا (use_vec_reduce) {
        dispatches.push_back({
            reduce_pipeline, std::move(reduce_params), std::move(reduce_entries), { (uint32_t) nrows, 1u }
        });
    }

    أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساكن webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                ggml_tensor *    Q,
                                                ggml_tensor *    K,
                                                ggml_tensor *    V,
                                                ggml_tensor *    ماسک,
                                                ggml_tensor *    sinks,
                                                ggml_tensor *    dst) {
    ggml_webgpu_flash_attn_op op = ggml_webgpu_flash_attn_prepare(ctx, Q, K, V, ماسک, sinks, dst);
    إذا (ggml_webgpu_flash_attn_use_vec_لار(ctx->global_ctx, Q, K, V)) {
        أرجع ggml_webgpu_flash_attn_vec(ctx, Q, K, V, ماسک, sinks, dst, std::move(op));
    }
    أرجع ggml_webgpu_flash_attn_مستقیم(ctx, op);
}

ساكن webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقي is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = تش_اشاره;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    خپلکاری *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
    ثابت منطقي inplace   = decisions->inplace;

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (src->nb[0] / ggml_type_اندازه(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),
                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2] };

    ggml_tensor * effective_src = src;
    إذا (is_unary) {
        ggml_unary_op unary_op = ggml_get_unary_op(dst);
        بدل (unary_op) {
            حالة GGML_UNARY_OP_XIELU:
                {
                    // Get float parameters and reinterpret their bit patterns as uint32_t
                    // for passing through the params buffer
                    عائم alpha_n = ggml_get_op_params_f32(dst, 1);
                    عائم alpha_p = ggml_get_op_params_f32(dst, 2);
                    عائم beta    = ggml_get_op_params_f32(dst, 3);
                    عائم eps     = ggml_get_op_params_f32(dst, 4);
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_n));
                    params.push_back(ggml_webgpu_u32_from_f32(alpha_p));
                    params.push_back(ggml_webgpu_u32_from_f32(beta));
                    params.push_back(ggml_webgpu_u32_from_f32(eps));
                    اكسر;
                }
            افتراضي:
                اكسر;
        }
    } وإلا إذا (dst->op == GGML_OP_CLAMP) {
        عائم clamp_min = ggml_get_op_params_f32(dst, 0);
        عائم clamp_max = ggml_get_op_params_f32(dst, 1);
        params.push_back(ggml_webgpu_u32_from_f32(clamp_min));
        params.push_back(ggml_webgpu_u32_from_f32(clamp_max));
    } وإلا إذا (dst->op == GGML_OP_FILL) {
        عائم fill_val = ggml_get_op_params_f32(dst, 0);
        params.push_back(ggml_webgpu_u32_from_f32(fill_val));
        effective_src = dst;  // fill simply fills dst
    }

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, effective_src),
    };
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_دوګونی_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقي      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_دوګونی_pipeline(shader_lib_ctx, src_overlap);
    خپلکاری *          decisions   = static_cast<ggml_webgpu_دوګونی_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_دننه_واټن = ggml_webgpu_tensor_align_دننه_واټن(ctx, src0);
    size_t src1_webgpu_tensor_align_دننه_واټن = ggml_webgpu_tensor_align_دننه_واټن(ctx, src1);

    uint32_t دننه_واټن_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t دننه_واټن_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    size_t   merged_دننه_واټن = 0;
    size_t   merged_اندازه   = 0;
    إذا (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_دننه_واټن = merged_range.دننه_واټن;
        merged_اندازه   = merged_range.size;
        دننه_واټن_src0   = ggml_webgpu_tensor_merged_element_دننه_واټن(src0, merged_range);
        دننه_واټن_src1   = ggml_webgpu_tensor_merged_element_دننه_واټن(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        دننه_واټن_src0,
        دننه_واټن_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src0->nb[0] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[0] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0],
        (uint32_t) src1->ne[1],
        (uint32_t) src1->ne[2],
        (uint32_t) src1->ne[3],
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries;

    إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src0), merged_دننه_واټن, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_دننه_واټن,
                                                            ggml_webgpu_tensor_binding_اندازه(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_دننه_واټن,
                                                            ggml_webgpu_tensor_binding_اندازه(ctx, src1)));
        إذا (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_جمع_کړه_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_جمع_کړه_id_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src2->nb[0] / ggml_type_اندازه(src2->type)),
        (uint32_t) (src2->nb[1] / ggml_type_اندازه(src2->type)),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, src2));

    إذا (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       ټول_wg       = ggml_nrows(dst);
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    compute_2d_workgroups(ټول_wg, max_wg_فی_dim, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_concat(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    dst) {
    uint32_t ne  = (uint32_t) ggml_nelements(dst);
    uint32_t dim = (uint32_t) dst->op_params[0];
    إذا (ggml_nbytes(src0) == 0 && ggml_nbytes(src1) == 0) {
        أرجع {};
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقي      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    خپلکاری *          decisions   = static_cast<ggml_webgpu_دوګونی_shader_decisions *>(pipeline.context.get());

    uint32_t دننه_واټن_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t دننه_واټن_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    size_t   merged_دننه_واټن = 0;
    size_t   merged_اندازه   = 0;
    إذا (decisions->src_overlap) {
        إذا (ggml_nbytes(src0) == 0) {
            merged_دننه_واټن = ggml_webgpu_tensor_align_دننه_واټن(ctx, src1);
            merged_اندازه   = ggml_webgpu_tensor_binding_اندازه(ctx, src1);
            دننه_واټن_src0   = 0;
            دننه_واټن_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
        } وإلا إذا (ggml_nbytes(src1) == 0) {
            merged_دننه_واټن = ggml_webgpu_tensor_align_دننه_واټن(ctx, src0);
            merged_اندازه   = ggml_webgpu_tensor_binding_اندازه(ctx, src0);
            دننه_واټن_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
            دننه_واټن_src1   = 0;
        } وإلا {
            ثابت ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_دننه_واټن = merged_range.دننه_واټن;
            merged_اندازه   = merged_range.size;
            دننه_واټن_src0   = ggml_webgpu_tensor_merged_element_دننه_واټن(src0, merged_range);
            دننه_واټن_src1   = ggml_webgpu_tensor_merged_element_دننه_واټن(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     دننه_واټن_src0,
                                     دننه_واټن_src1,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src1->nb[0] / ggml_type_اندازه(src1->type)),
                                     (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
                                     (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
                                     (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)),
                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     dim,
                                     (uint32_t) src0->ne[dim] };

    std::vector<wgpu::Bind_ډله_ثبت> entries = {};
    إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src0), merged_دننه_واټن, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = { ne,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) /
                                                 ggml_type_اندازه(src0->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (src0->nb[0] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
                                     (uint32_t) (src0->ne[0]),
                                     (uint32_t) (src0->ne[1]),
                                     (uint32_t) (src0->ne[2]),
                                     (uint32_t) (src0->ne[3]),
                                     (uint32_t) (dst->ne[0]),
                                     (uint32_t) (dst->ne[1]),
                                     (uint32_t) (dst->ne[2]) };

    std::vector<wgpu::Bind_ډله_ثبت> entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    خپلکاری *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
                                                                 ggml_tensor *    rn_src,
                                                                 ggml_tensor *    rn_dst,
                                                                 ggml_tensor *    mul_src0,
                                                                 ggml_tensor *    mul_src1,
                                                                 ggml_tensor *    dst) {
    ggml_tensor * mul_src;

    إذا (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
        mul_src = mul_src1;
    } وإلا إذا (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
        mul_src = mul_src0;
    } وإلا {
        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
    }

    uint32_t دننه_واټن_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_اندازه(rn_src->type));
    uint32_t دننه_واټن_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_اندازه(mul_src->type));
    size_t merged_دننه_واټن = 0;
    size_t merged_اندازه   = 0;

    std::vector<uint32_t> params = {
        دننه_واټن_rn_src,
        دننه_واټن_mul_src,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (rn_src->nb[1] / ggml_type_اندازه(rn_src->type)),
        (uint32_t) (rn_src->nb[2] / ggml_type_اندازه(rn_src->type)),
        (uint32_t) (rn_src->nb[3] / ggml_type_اندازه(rn_src->type)),
        (uint32_t) (mul_src->nb[1] / ggml_type_اندازه(mul_src->type)),
        (uint32_t) (mul_src->nb[2] / ggml_type_اندازه(mul_src->type)),
        (uint32_t) (mul_src->nb[3] / ggml_type_اندازه(mul_src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
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

    std::vector<wgpu::Bind_ډله_ثبت> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقي      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    خپلکاری *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    إذا (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_دننه_واټن  = merged_range.دننه_واټن;
        merged_اندازه    = merged_range.size;
        دننه_واټن_rn_src  = ggml_webgpu_tensor_merged_element_دننه_واټن(rn_src, merged_range);
        دننه_واټن_mul_src = ggml_webgpu_tensor_merged_element_دننه_واټن(mul_src, merged_range);
        params[0]      = دننه_واټن_rn_src;
        params[1]      = دننه_واټن_mul_src;
    }

    إذا (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, mul_src));
    } وإلا إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(rn_src), merged_دننه_واټن, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x,
                          wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_لیکه_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        (uint32_t) src->ne[3],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_لیکه_norm_pipeline(shader_lib_ctx);
    خپلکاری *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src) };
    إذا (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x,
                          wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    src2,
                                          ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    خپلکاری * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت منطقي inplace         = decisions->inplace;
    ثابت صحيح  has_freq_ضریب = (src2 != تش_اشاره);

    ثابت صحيح n_dims     = ((int32_t *) dst->op_params)[1];
    ثابت صحيح mode       = ((int32_t *) dst->op_params)[2];
    ثابت صحيح n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ثابت صحيح n_offs     = ((int32_t *) dst->op_params)[15];

    عائم freq_بنسټ;
    عائم freq_scale;
    عائم ext_ضریب;
    عائم attn_ضریب;
    عائم beta_fast;
    عائم beta_slow;
    حافظه_نقل(&freq_بنسټ, (int32_t *) dst->op_params + 5, حجم(عائم));
    حافظه_نقل(&freq_scale, (int32_t *) dst->op_params + 6, حجم(عائم));
    حافظه_نقل(&ext_ضریب, (int32_t *) dst->op_params + 7, حجم(عائم));
    حافظه_نقل(&attn_ضریب, (int32_t *) dst->op_params + 8, حجم(عائم));
    حافظه_نقل(&beta_fast, (int32_t *) dst->op_params + 9, حجم(عائم));
    حافظه_نقل(&beta_slow, (int32_t *) dst->op_params + 10, حجم(عائم));

    صحيح sections[4];
    حافظه_نقل(sections, (int32_t *) dst->op_params + 11, 4 * حجم(صحيح));

    عائم theta_scale = powf(freq_بنسټ, -2.0f / n_dims);

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنسټ, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        src2 != تش_اشاره ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
        (uint32_t) ggml_nelements(src0) / 2,
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        (uint32_t) n_dims,
        (uint32_t) mode,
        ggml_webgpu_u32_from_f32(theta_scale),
        ggml_webgpu_u32_from_f32(attn_ضریب),
        ggml_webgpu_u32_from_f32(freq_scale),
        ggml_webgpu_u32_from_f32(ext_ضریب),
        ggml_webgpu_u32_from_f32(corr_dims[0]),
        ggml_webgpu_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0],
        (uint32_t) sections[1],
        (uint32_t) sections[2],
        (uint32_t) sections[3],
        (uint32_t) n_offs
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries     = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    إذا (has_freq_ضریب) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 2, src2));
    }
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    ثابت منطقي      src_overlap = src1 != تش_اشاره && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    خپلکاری * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ثابت صحيح وېشل = (src1 != تش_اشاره);

    uint32_t دننه_واټن_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t دننه_واټن_src1 =
        src1 != تش_اشاره ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)) : 0;
    size_t merged_دننه_واټن = 0;
    size_t merged_اندازه   = 0;
    إذا (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_دننه_واټن = merged_range.دننه_واټن;
        merged_اندازه   = merged_range.size;
        دننه_واټن_src0   = ggml_webgpu_tensor_merged_element_دننه_واټن(src0, merged_range);
        دننه_واټن_src1   = ggml_webgpu_tensor_merged_element_دننه_واټن(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        دننه_واټن_src0,
        دننه_واټن_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
        src1 != تش_اشاره ? (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        src1 != تش_اشاره ? (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        src1 != تش_اشاره ? (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)) :
                          (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) dst->ne[0],
        (uint32_t) dst->ne[1],
        (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],                // swapped
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 2)),  // alpha, for swiglu_oai
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 3)),  //حد
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries;
    uint32_t                          dst_binding = 1;
    إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_ډله_ثبت(0, ggml_webgpu_tensor_buf(src0), merged_دننه_واټن, merged_اندازه));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src0));
    }
    إذا (وېشل && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = تش_اشاره;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    خپلکاری *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    // params unchanged
    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
        (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
        (uint32_t) ggml_nelements(dst),
        (uint32_t) src->ne[0],
        (uint32_t) src->ne[1],
        (uint32_t) src->ne[2],
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 1))   // bias
    };

    // bindgroups unchanged
    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src) };

    إذا (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
                                              ggml_tensor *    src0,
                                              ggml_tensor *    src1,
                                              ggml_tensor *    src2,
                                              ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    خپلکاری *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت منطقي inplace     = decisions->inplace;
    ثابت صحيح  has_ماسک    = (src1 != تش_اشاره);
    ثابت صحيح  has_sink    = (src2 != تش_اشاره);
    عائم      max_bias    = ggml_get_op_params_f32(dst, 1);
    عائم      n_سر_log2 = عائم(1u << (uint32_t) floor(log2(src0->ne[2])));
    عائم      m0          = powf(2.0f, -(max_bias) / n_سر_log2);
    عائم      m1          = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        has_ماسک ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)) : 0,
        has_sink ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)) : 0,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
        has_ماسک ? (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)) : 0,
        has_ماسک ? (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)) : 0,
        has_ماسک ? (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)) : 0,
        (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
        (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        (uint32_t) src0->ne[2],
        has_ماسک ? (uint32_t) src1->ne[2] : 0,
        has_ماسک ? (uint32_t) src1->ne[3] : 0,
        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0)),  // scale
        ggml_webgpu_u32_from_f32(max_bias),
        ggml_webgpu_u32_from_f32(n_سر_log2),
        ggml_webgpu_u32_from_f32(m0),
        ggml_webgpu_u32_from_f32(m1)
    };

    std::vector<wgpu::Bind_ډله_ثبت> entries     = { ggml_webgpu_make_bind_ډله_ثبت(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_دننه_واټن(ctx, src0),
        ggml_webgpu_tensor_binding_اندازه(ctx, src0)) };
    uint32_t                          binding_عدد = 1;
    إذا (has_ماسک) {
        entries.push_back(ggml_webgpu_make_bind_ډله_ثبت(binding_عدد, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_دننه_واټن(ctx, src1),
                                                            ggml_webgpu_tensor_binding_اندازه(ctx, src1)));
        binding_عدد++;
    }
    إذا (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد, src2));
        binding_عدد++;
    }
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, binding_عدد, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = ggml_nrows(dst);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

ساكن webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقي is_سر_k = dst->op == GGML_OP_سر_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = تش_اشاره;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    shader_lib_ctx.wg_mem_حد_بایټونه = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_اندازه;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    خپلکاری * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_یوځای_pipeline = ctx->shader_lib->get_argsort_یوځای_pipeline(shader_lib_ctx);

    ثابت uint32_t src_ne0 = (uint32_t) src->ne[0];
    ثابت uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ثابت uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_اندازه);
    ثابت uint32_t بلاک_اندازه_3 =
        is_سر_k ? std::min(argsort_decisions->wg_اندازه, (uint32_t) dst->ne[0]) : argsort_decisions->wg_اندازه;
    uint32_t بهر_ne0 = src_ne0;
    إذا (is_سر_k) {
        إذا (npr > 1) {
            ثابت uint32_t وروستی_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_اندازه;
            بهر_ne0                  = (npr - 1) * بلاک_اندازه_3 + std::min(وروستی_tile, بلاک_اندازه_3);
        } وإلا {
            بهر_ne0 = بلاک_اندازه_3;
        }
    }

    uint32_t یوځای_len    = بلاک_اندازه_3;
    uint32_t یوځای_passes = 0;
    طالما (یوځای_len < بهر_ne0) {
        یوځای_len <<= 1;
        یوځای_passes++;
    }

    ثابت منطقي پیل_دننه_tmp = (یوځای_passes % 2) == 1;

    ثابت size_t dst_دننه_واټن = ggml_webgpu_tensor_دننه_واټن(dst);
    ثابت size_t idx_nbytes = بهر_ne0 * ggml_nrows(dst) * حجم(int32_t);
    ثابت size_t tmp_دننه_واټن =
        ROUNDUP_POW2(dst_دننه_واټن + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment);
    ثابت size_t tmp_binding_اندازه = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ثابت size_t dst_binding_اندازه =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ثابت uint32_t دننه_واټن_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type));
    ثابت uint32_t دننه_واټن_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type));
    ثابت uint32_t دننه_واټن_tmp  = 0;
    ثابت uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_idx1 = بهر_ne0;
    ثابت uint32_t stride_idx2 = بهر_ne0 * (uint32_t) dst->ne[1];
    ثابت uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ثابت uint32_t init_دننه_واټن       = پیل_دننه_tmp ? دننه_واټن_tmp : دننه_واټن_dst;
    ثابت size_t   init_align_دننه_واټن = پیل_دننه_tmp ? tmp_دننه_واټن : ggml_webgpu_tensor_align_دننه_واټن(ctx, dst);
    ثابت size_t   init_binding_اندازه = پیل_دننه_tmp ? tmp_binding_اندازه : dst_binding_اندازه;

    std::vector<uint32_t> init_params = {
        دننه_واټن_src,  init_دننه_واټن, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], بهر_ne0,
        بلاک_اندازه_3,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ثابت uint32_t ټول_wg_init  = npr * nrows;
    ثابت uint32_t max_wg_فی_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    compute_2d_workgroups(ټول_wg_init, max_wg_فی_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::Bind_ډله_ثبت> init_entries = {
        ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
        ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(dst), init_align_دننه_واټن, init_binding_اندازه)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    إذا (یوځای_passes == 0) {
        أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    منطقي     دننه_is_tmp = پیل_دننه_tmp;
    uint32_t len       = بلاک_اندازه_3;
    طالما (len < بهر_ne0) {
        ثابت uint32_t nm = CEIL_DIV(بهر_ne0, 2 * len);

        ثابت منطقي     بهر_is_tmp  = !دننه_is_tmp;
        ثابت uint32_t دننه_واټن_دننه   = دننه_is_tmp ? دننه_واټن_tmp : دننه_واټن_dst;
        ثابت uint32_t دننه_واټن_بهر  = بهر_is_tmp ? دننه_واټن_tmp : دننه_واټن_dst;
        ثابت size_t   align_دننه    = دننه_is_tmp ? tmp_دننه_واټن : ggml_webgpu_tensor_align_دننه_واټن(ctx, dst);
        ثابت size_t   align_بهر   = بهر_is_tmp ? tmp_دننه_واټن : ggml_webgpu_tensor_align_دننه_واټن(ctx, dst);
        ثابت size_t   اندازه_دننه     = دننه_is_tmp ? tmp_binding_اندازه : dst_binding_اندازه;
        ثابت size_t   اندازه_بهر    = بهر_is_tmp ? tmp_binding_اندازه : dst_binding_اندازه;
        ثابت uint32_t سر_k_بهر   = (is_سر_k && nm == 1) ? (uint32_t) dst->ne[0] : بهر_ne0;
        ثابت uint32_t stride_out1 = سر_k_بهر;
        ثابت uint32_t stride_out2 = سر_k_بهر * (uint32_t) dst->ne[1];
        ثابت uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> یوځای_params = { دننه_واټن_src,
                                               دننه_واټن_دننه,
                                               دننه_واټن_بهر,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               بهر_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               سر_k_بهر,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::Bind_ډله_ثبت> یوځای_entries = {
            ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
            ggml_webgpu_make_bind_ډله_ثبت(1, ggml_webgpu_tensor_buf(dst), align_دننه, اندازه_دننه),
            ggml_webgpu_make_bind_ډله_ثبت(2, ggml_webgpu_tensor_buf(dst), align_بهر, اندازه_بهر)
        };

        uint32_t       wg_x_یوځای;
        uint32_t       wg_y_یوځای;
        ثابت uint32_t ټول_wg_یوځای = nm * nrows;
        compute_2d_workgroups(ټول_wg_یوځای, max_wg_فی_dim, wg_x_یوځای, wg_y_یوځای);

        dispatches.push_back({
            argsort_یوځای_pipeline, std::move(یوځای_params), std::move(یوځای_entries), { wg_x_یوځای, wg_y_یوځای }
        });

        len <<= 1;
        دننه_is_tmp = !دننه_is_tmp;
    }

    أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساكن webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = تش_اشاره;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        ټول_wg = ggml_nrows(dst);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_مجموعه_لیکې(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقي                  ټول_مجموعه = dst->op == GGML_OP_مجموعه;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     ټول_مجموعه ? 0 : (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
                                     ټول_مجموعه ? 0 : (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
                                     ټول_مجموعه ? 0 : (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),
                                     ټول_مجموعه ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     ټول_مجموعه ? 1 : (uint32_t) src->ne[1],
                                     ټول_مجموعه ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_مجموعه_لیکې_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = ټول_مجموعه ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن منطقي ggml_webgpu_can_fuse_rms_norm_mul(ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx) {
    إذا (!ggml_can_fuse(cgraph, غوټه_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        أرجع خطأ;
    }

    // additional constraints specific to this fusion
    ثابت ggml_tensor * rms_norm = cgraph->nodes[غوټه_idx];
    ثابت ggml_tensor * mul      = cgraph->nodes[غوټه_idx + 1];

    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    // rms_norm only supports f32
    إذا (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    // if rms_norm is the B operand, then we don't handle broadcast
    إذا (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        أرجع خطأ;
    }
    // rms_norm shader assumes contiguous rows
    إذا (!ggml_is_contiguous_لیکې(mul->src[0]) || !ggml_is_contiguous_لیکې(mul->src[1])) {
        أرجع خطأ;
    }

    أرجع ریښتیا;
}

ساكن منطقي ggml_webgpu_can_fuse_gdn_cache(ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, صحيح & عدد_encoded_ops) {
    ثابت ggml_tensor * gdn = cgraph->nodes[غوټه_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    إذا (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * src_v     = gdn->src[2];
    ثابت int64_t       S_v       = src_v->ne[0];
    ثابت int64_t       H         = src_v->ne[1];
    ثابت int64_t       n_نښې  = src_v->ne[2];
    ثابت int64_t       n_seqs    = src_v->ne[3];
    ثابت int64_t       D         = S_v * S_v * H;
    ثابت int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    ثابت int64_t       n_written = std::min<int64_t>(n_نښې, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ثابت size_t لکۍ_off = ggml_لیکه_اندازه(GGML_TYPE_F32, S_v * H * n_نښې * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ثابت ggml_tensor * cpy     = تش_اشاره;
    صحيح                 cpy_idx = 0;
    لكل (صحيح j = غوټه_idx + 1; j < cgraph->n_nodes && cpy == تش_اشاره; ++j) {
        ثابت ggml_tensor * n = cgraph->nodes[j];
        إذا (ggml_op_is_تشه(n->op) || ggml_is_تشه(n)) {
            استمر;
        }
        إذا (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            أرجع خطأ;
        }
        cpy     = n;
        cpy_idx = j;
    }
    إذا (cpy == تش_اشاره) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ثابت ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    إذا (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != لکۍ_off ||
        !ggml_is_contiguous(cpy_src)) {
        أرجع خطأ;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ثابت std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    إذا (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->معلومات == تش_اشاره ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_اندازه(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_لیکه_اندازه(GGML_TYPE_F32, D)) {
        أرجع خطأ;
    }

    عدد_encoded_ops = cpy_idx - غوټه_idx + 1;

    أرجع ریښتیا;
}

ساكن webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ثابت uint32_t        ډېرپېښ_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),

                                     (uint32_t) (src->nb[0] / ggml_type_اندازه(src->type)),
                                     (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
                                     (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
                                     (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),

                                     (uint32_t) (dst->nb[0] / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (dst->nb[1] / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (dst->nb[2] / ggml_type_اندازه(dst->type)),
                                     (uint32_t) (dst->nb[3] / ggml_type_اندازه(dst->type)),

                                     (uint32_t) src->ne[0],
                                     (uint32_t) src->ne[1],
                                     (uint32_t) src->ne[2],
                                     (uint32_t) src->ne[3],

                                     (uint32_t) dst->ne[0],
                                     (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2],
                                     (uint32_t) dst->ne[3],

                                     ډېرپېښ_flags };

    std::vector<wgpu::Bind_ډله_ثبت> entries = { ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_ډله_ثبت(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    خپلکاری *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t ټول_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(ټول_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
ساكن std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           صحيح            غوټه_idx,
                                                           صحيح &          عدد_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  غوټه  = nodes[غوټه_idx];

    إذا (ggml_is_تشه(غوټه)) {
        أرجع std::nullopt;
    }
    إذا ((غوټه->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        أرجع std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << غوټه << ", " << ggml_op_name(غوټه->op) << ")");

    ggml_tensor * src0 = غوټه->src[0];
    ggml_tensor * src1 = غوټه->src[1];
    ggml_tensor * src2 = غوټه->src[2];

    بدل (غوټه->op) {
            // no-ops
        حالة GGML_OP_NONE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلول:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE:
            أرجع std::nullopt;
        حالة GGML_OP_DUP:
        حالة GGML_OP_CPY:
        حالة GGML_OP_CONT:
            أرجع ggml_webgpu_cpy(ctx, src0, غوټه);
        حالة GGML_OP_سېټ:
            أرجع ggml_webgpu_سېټ(ctx, src0, src1, غوټه);
        حالة GGML_OP_سېټ_لیکې:
            أرجع ggml_webgpu_سېټ_لیکې(ctx, src0, src1, غوټه);
        حالة GGML_OP_GET_لیکې:
            أرجع ggml_webgpu_get_لیکې(ctx, src0, src1, غوټه);
        حالة GGML_OP_MUL_MAT:
            أرجع ggml_webgpu_mul_mat(ctx, src0, src1, غوټه);
        حالة GGML_OP_MUL_MAT_ID:
            أرجع ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, غوټه);
        حالة GGML_OP_FLASH_ATTN_EXT:
            أرجع ggml_webgpu_flash_attn(ctx, src0, src1, src2, غوټه->src[3], غوټه->src[4], غوټه);
        حالة GGML_OP_جمع_کړه:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
            أرجع ggml_webgpu_دوګونی_op(ctx, src0, src1, غوټه);
        حالة GGML_OP_جمع_کړه_ID:
            أرجع ggml_webgpu_جمع_کړه_id(ctx, src0, src1, src2, غوټه);
        حالة GGML_OP_CONCAT:
            أرجع ggml_webgpu_concat(ctx, src0, src1, غوټه);
        حالة GGML_OP_REPEAT:
            أرجع ggml_webgpu_repeat(ctx, src0, غوټه);
        حالة GGML_OP_RMS_NORM:
            إذا (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, غوټه_idx)) {
                عدد_encoded_ops        = 2;
                ggml_tensor * mul_غوټه = nodes[غوټه_idx + 1];
                أرجع ggml_webgpu_rms_norm_mul(ctx, src0, غوټه, mul_غوټه->src[0], mul_غوټه->src[1], mul_غوټه);
            } وإلا {
                أرجع ggml_webgpu_لیکه_norm(ctx, src0, غوټه);
            }
        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
            أرجع ggml_webgpu_لیکه_norm(ctx, src0, غوټه);
        حالة GGML_OP_ROPE:
            أرجع ggml_webgpu_rope(ctx, src0, src1, src2, غوټه);
        حالة GGML_OP_GLU:
            أرجع ggml_webgpu_glu(ctx, src0, src1, غوټه);
        حالة GGML_OP_SCALE:
            أرجع ggml_webgpu_scale(ctx, src0, غوټه);
        حالة GGML_OP_SOFT_MAX:
            أرجع ggml_webgpu_soft_max(ctx, src0, src1, src2, غوټه);
        حالة GGML_OP_UNARY:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_FILL:
        حالة GGML_OP_LOG:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_SIN:
        حالة GGML_OP_COS:
        حالة GGML_OP_DIAG:
        حالة GGML_OP_TRI:
            أرجع ggml_webgpu_unary_op(ctx, src0, غوټه);
        حالة GGML_OP_SOLVE_TRI:
            أرجع ggml_webgpu_solve_tri(ctx, src0, src1, غوټه);
        حالة GGML_OP_SSM_CONV:
            أرجع ggml_webgpu_ssm_conv(ctx, src0, src1, غوټه);
        حالة GGML_OP_SSM_لټون:
            أرجع ggml_webgpu_ssm_لټون(ctx, src0, src1, src2, غوټه->src[3], غوټه->src[4], غوټه->src[5], غوټه->src[6],
                                        غوټه);
        حالة GGML_OP_GATED_DELTA_NET:
            إذا (ggml_webgpu_can_fuse_gdn_cache(cgraph, غوټه_idx, عدد_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[غوټه_idx + عدد_encoded_ops - 1]->src[1];
                أرجع ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, غوټه->src[3], غوټه->src[4], غوټه->src[5],
                                                   غوټه, dst_fuse);
            } وإلا {
                أرجع ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, غوټه->src[3], غوټه->src[4], غوټه->src[5],
                                                   غوټه, تش_اشاره);
            }
        حالة GGML_OP_PAD:
            أرجع ggml_webgpu_pad(ctx, src0, غوټه);
        حالة GGML_OP_ARGMAX:
            أرجع ggml_webgpu_argmax(ctx, src0, غوټه);
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_سر_K:
            // we reuse the same argsort implementation for top_k
            أرجع ggml_webgpu_argsort(ctx, src0, غوټه);
        حالة GGML_OP_CUMSUM:
            أرجع ggml_webgpu_cumsum(ctx, src0, غوټه);
        حالة GGML_OP_مجموعه:
        حالة GGML_OP_مجموعه_لیکې:
            أرجع ggml_webgpu_مجموعه_لیکې(ctx, src0, غوټه);
        حالة GGML_OP_CONV_2D:
            أرجع ggml_webgpu_conv_2d(ctx, src0, src1, غوټه);
        حالة GGML_OP_CONV_2D_DW:
            أرجع ggml_webgpu_conv_2d_dw(ctx, src0, src1, غوټه);
        حالة GGML_OP_IM2_ستون:
            أرجع ggml_webgpu_im2col(ctx, src0, src1, غوټه);
        حالة GGML_OP_UPSCALE:
            أرجع ggml_webgpu_upscale(ctx, src0, غوټه);
        افتراضي:
            أرجع std::nullopt;
    }
}

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
ساكن فراغ ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ثابت std::vector<std::string> & pipeline_نومونه,
                                                        uint32_t &                       عدد_inflight_batches) {
    إذا (pipeline_نومونه.تشه()) {
        أرجع;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Resolve_Query_سېټ(ctx->profile_timestamp_query_سېټ, 0, ctx->profile_timestamp_query_شمېر,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_بفر_منزل_بفر(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_شمېر * حجم(uint64_t));

    wgpu::Command_بفر profile_commands = encoder.پای();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, عدد_inflight_batches);

    ثابت size_t mapped_اندازه = ctx->profile_timestamp_query_شمېر * حجم(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_شمېر == 2 * pipeline_نومونه.size());

    ggml_backend_webgpu_map_بفر(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::Map_ډېرپېښ::Read, 0,
                                   mapped_اندازه);
    ثابت uint64_t * ts_معلومات = (ثابت uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_اندازه);

    لكل (size_t i = 0; i < pipeline_نومونه.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ثابت مزدوج elapsed_ms = مزدوج(ts_معلومات[2 * i + 1] - ts_معلومات[2 * i]) * 1e-6;
        ctx->shader_gpu_وخت_ms[pipeline_نومونه[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#د_که_پای

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
ساكن فراغ ggml_backend_webgpu_check_سېټ_لیکې(webgpu_context & ctx, uint32_t & عدد_inflight_batches) {
#که_تعریف_شوی GGML_WEBGPU_CHECK_سېټ_لیکې
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_بفر_منزل_بفر(ctx->سېټ_لیکې_dev_error_buf, 0, ctx->سېټ_لیکې_host_error_buf, 0,
                               ctx->سېټ_لیکې_host_error_buf.Get_اندازه());
    wgpu::Command_بفر commands = encoder.پای();
    ggml_backend_webgpu_submit_commands(ctx, commands, عدد_inflight_batches);
    ggml_backend_webgpu_map_بفر(ctx->global_ctx, ctx->سېټ_لیکې_host_error_buf, wgpu::Map_ډېرپېښ::Read, 0,
                                   ctx->سېټ_لیکې_host_error_buf.Get_اندازه());
    ثابت uint32_t * error_معلومات = (ثابت uint32_t *) ctx->سېټ_لیکې_host_error_buf.GetConstMappedRange();
    إذا (*error_معلومات) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->سېټ_لیکې_host_error_buf.Unmap();
#وإلا
    GGML_ناکارول_شوی(ctx);
    GGML_ناکارول_شوی(عدد_inflight_batches);
#د_که_پای
}

ساكن ggml_حالت ggml_backend_webgpu_ګراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_ټول_پیل(ګراف_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t عدد_batched_kernels  = 0;
    uint32_t عدد_inflight_batches = 0;
    منطقي     لري_سېټ_لیکې    = خطأ;
    صحيح      عدد_encoded_ops      = 1;
    صحيح      غوټه_idx             = 0;

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_شمېر = 0;
    std::vector<std::string> profile_pipeline_نومونه;
#د_که_پای

    ctx->فعال_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    إذا (ctx->batch_compute_passes) {
        ctx->فعال_compute_pass = ctx->فعال_command_encoder.BeginComputePass();
    }

    طالما (غوټه_idx < cgraph->n_nodes) {
        إذا (cgraph->nodes[غوټه_idx]->op == GGML_OP_سېټ_لیکې) {
            لري_سېټ_لیکې = ریښتیا;
        }
        إذا (خپلکاری cmd = ggml_webgpu_encode(ctx, cgraph, غوټه_idx, عدد_encoded_ops)) {
            commands.push_back(*cmd);
            عدد_batched_kernels += cmd.value().عدد_kernels;
#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_نومونه.insert(profile_pipeline_نومونه.end(), cmd->pipeline_نومونه.begin(),
                                          cmd->pipeline_نومونه.end());
#د_که_پای
        }

        إذا (عدد_batched_kernels >= ctx->global_ctx->command_submit_batch_اندازه) {
            إذا (ctx->فعال_compute_pass) {
                ctx->فعال_compute_pass.پای_ټکی_2();
            }
            عدد_batched_kernels                = 0;
            wgpu::Command_بفر batch_commands = ctx->فعال_command_encoder.پای();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, عدد_inflight_batches);

            // reset state for next batch
            ctx->فعال_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            إذا (ctx->batch_compute_passes) {
                ctx->فعال_compute_pass = ctx->فعال_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            إذا (ctx->profile_timestamp_query_شمېر + 2 * ctx->global_ctx->command_submit_batch_اندازه >=
                WEBGPU_MAX_PROFILE_QUERY_شمېر) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_نومونه, عدد_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_شمېر = 0;
                profile_pipeline_نومونه.clear();
            }
#د_که_پای
        }

        غوټه_idx += عدد_encoded_ops;
        عدد_encoded_ops = 1;
    }

    إذا (ctx->فعال_compute_pass) {
        ctx->فعال_compute_pass.پای_ټکی_2();
        ctx->فعال_compute_pass = تش_اشاره;
    }

    إذا (عدد_batched_kernels > 0) {
        wgpu::Command_بفر batch_commands = ctx->فعال_command_encoder.پای();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, عدد_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->فعال_command_encoder = تش_اشاره;

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_نومونه, عدد_inflight_batches);
#د_که_پای

    إذا (لري_سېټ_لیکې) {
        ggml_backend_webgpu_check_سېټ_لیکې(ctx, عدد_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_ټول_پای_ټکی(ګراف_compute, ctx->global_ctx);
    أرجع GGML_حالت_SUCCESS;
}

بنية ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    منطقي                  recorded = خطأ;
};

ساكن ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    خپلکاری * event_ctx      = جديد ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    خپلکاری * event   = جديد ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    أرجع event;
}

ساكن فراغ ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_ناکارول_شوی(dev);
    ړنګول static_cast<ggml_backend_webgpu_event_context *>(event->context);
    ړنګول event;
}

ساكن فراغ ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_ناکارول_شوی(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    إذا (!event_ctx->recorded) {
        أرجع;
    }
    wgpu::Wait_حالت حالت =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    إذا (حالت == wgpu::Wait_حالت::Timed_بهر) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = خطأ;
}

ساكن فراغ ggml_backend_webgpu_event_ریکارډ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.On_Submitted_Work_بشپړ(
        wgpu::Callback_ډېرپېښ::AllowSpontaneous, [](wgpu::کتار_Work_بشپړ_حالت, wgpu::StringView) {});
    event_ctx->recorded = ریښتیا;
}

ساكن فراغ ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ناکارول_شوی(backend);
    ggml_backend_webgpu_device_event_synchronize(تش_اشاره, event);
}

ساكن فراغ ggml_backend_webgpu_سېټ_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ثابت فراغ *   معلومات,
                                                 size_t         دننه_واټن,
                                                 size_t         size) {
    GGML_ناکارول_شوی(backend);
    خپلکاری * buf_ctx      = (ggml_backend_webgpu_بفر_context *) tensor->بفر->context;
    size_t ټول_دننه_واټن = ggml_webgpu_tensor_دننه_واټن(tensor) + دننه_واټن;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_بفر(buf_ctx->بفر, ټول_دننه_واټن, معلومات, (size / 4) * 4);

    إذا (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t پاتې_شوی_اندازه = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        لكل (size_t i = 0; i < پاتې_شوی_اندازه; i++) {
            ((uint8_t *) &val32)[i] = ((ثابت uint8_t *) معلومات)[size - پاتې_شوی_اندازه + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, val32,
                                          ټول_دننه_واټن + (size - پاتې_شوی_اندازه), پاتې_شوی_اندازه);
    }
}

ساكن فراغ ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_کتار(backend_ctx->webgpu_ctx->global_ctx);
}

ساكن ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_سېټ_tensor_async,
    /* .get_tensor_async        = */ عدم,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_webgpu_ګراف_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_ریکارډ,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ عدم,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

ساكن فراغ ggml_backend_webgpu_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_webgpu_بفر_context * ctx = static_cast<ggml_backend_webgpu_بفر_context *>(بفر->context);
    إذا (ctx != تش_اشاره && ctx->بفر != تش_اشاره) {
        ctx->بفر.Destroy();
        ړنګول ctx;
    }
}

// Returns the "fake" base pointer.
ساكن فراغ * ggml_backend_webgpu_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    GGML_ناکارول_شوی(بفر);
    أرجع webgpu_ptr_بنسټ;
}

ساكن فراغ ggml_backend_webgpu_بفر_memset_tensor(ggml_backend_بفر_t بفر,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                دننه_واټن,
                                                     size_t                size) {
    إذا (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        أرجع;
    }

    WEBGPU_CPU_PROFILE_ټول_پیل(memset_tensor);

    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << دننه_واټن << ", " << size << ")");

    size_t ټول_دننه_واټن = ggml_webgpu_tensor_دننه_واټن(tensor) + دننه_واټن;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, val32, ټول_دننه_واټن, size);
    WEBGPU_CPU_PROFILE_ټول_پای_ټکی(memset_tensor, buf_ctx->global_ctx);
}

ساكن فراغ ggml_backend_webgpu_بفر_سېټ_tensor(ggml_backend_بفر_t بفر,
                                                  ggml_tensor *         tensor,
                                                  ثابت فراغ *          معلومات,
                                                  size_t                دننه_واټن,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_ټول_پیل(سېټ_tensor);
    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << معلومات
                                                              << ", " << دننه_واټن << ", " << size << ")");

    size_t ټول_دننه_واټن = ggml_webgpu_tensor_دننه_واټن(tensor) + دننه_واټن;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    إذا (ټول_دننه_واټن % 4 != 0) {
        size_t lane = ټول_دننه_واټن % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t سر = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t سر_val = 0;
        لكل (size_t i = 0; i < سر; i++) {
            ((uint8_t *) &سر_val)[lane + i] = ((ثابت uint8_t *) معلومات)[i];
        }
        ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, سر_val, ټول_دننه_واټن, سر);

        ټول_دننه_واټن += سر;
        size -= سر;
        معلومات = (ثابت uint8_t *) معلومات + سر;
    }

    إذا (size > 0) {
        buf_ctx->global_ctx->queue.Write_بفر(buf_ctx->بفر, ټول_دننه_واټن, معلومات, (size / 4) * 4);
    }

    إذا (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t پاتې_شوی_اندازه = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        لكل (size_t i = 0; i < پاتې_شوی_اندازه; i++) {
            ((uint8_t *) &val32)[i] = ((ثابت uint8_t *) معلومات)[size - پاتې_شوی_اندازه + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, val32,
                                          ټول_دننه_واټن + (size - پاتې_شوی_اندازه), پاتې_شوی_اندازه);
    }
    WEBGPU_CPU_PROFILE_ټول_پای_ټکی(سېټ_tensor, buf_ctx->global_ctx);
}

ساكن فراغ ggml_backend_webgpu_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                  ثابت ggml_tensor *   tensor,
                                                  فراغ *                معلومات,
                                                  size_t                دننه_واټن,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_ټول_پیل(get_tensor);
    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << معلومات
                                                              << ", " << دننه_واټن << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t ټول_دننه_واټن = ggml_webgpu_tensor_دننه_واټن(tensor) + دننه_واټن;

    size_t local_دننه_واټن = ټول_دننه_واټن % 4;
    إذا (local_دننه_واټن != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        ټول_دننه_واټن -= local_دننه_واټن;
    }

    size_t final_اندازه = size + local_دننه_واټن;
    إذا (final_اندازه % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_اندازه += 4 - (final_اندازه % 4);
    }

    std::lock_guard<std::recursive_mutex> قلف(buf_ctx->global_ctx->mutex);

    إذا (buf_ctx->global_ctx->get_tensor_staging_buf == تش_اشاره ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_اندازه() < final_اندازه) {
        // Create a new staging buffer if it doesn't exist or is too small
        إذا (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_بفر(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_اندازه,
                                  wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_بفر_منزل_بفر(buf_ctx->بفر, ټول_دننه_واټن, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_اندازه);
    wgpu::Command_بفر commands = encoder.پای();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_بفر(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::Map_ډېرپېښ::Read, 0, final_اندازه);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ثابت فراغ * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_اندازه);

    // Copy the data from the mapped range to the output buffer
    std::حافظه_نقل(معلومات, (ثابت فراغ *) ((ثابت حرف *) mapped_range + local_دننه_واټن), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_ټول_پای_ټکی(get_tensor, buf_ctx->global_ctx);
}

ساكن فراغ ggml_backend_webgpu_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << بفر << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_ټول_پیل(clear);
    ggml_backend_webgpu_بفر_context * buf_ctx = (ggml_backend_webgpu_بفر_context *) بفر->context;
    ggml_backend_webgpu_بفر_memset(buf_ctx->global_ctx, buf_ctx->بفر, value, 0, بفر->size);
    WEBGPU_CPU_PROFILE_ټول_پای_ټکی(clear, buf_ctx->global_ctx);
}

ساكن ggml_backend_بفر_i ggml_backend_webgpu_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_webgpu_بفر_get_بنسټ,
    /* .init_tensor     = */ عدم,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ عدم,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_بفر_clear,
    /* .reset           = */ عدم,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

ساكن ثابت حرف * ggml_backend_webgpu_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    أرجع ctx->device_name.c_str();
}

ساكن ggml_backend_بفر_t ggml_backend_webgpu_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                          size_t                     size) {
    ساكن std::atomic<صحيح> بفر_شمېر;
    صحيح                     بفر_id = بفر_شمېر++;
    std::string             buf_name  = "tensor_buf" + std::to_string(بفر_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << بفر_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::بفر_2                         buf;
    ggml_webgpu_create_بفر(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::بفر_Usage::Storage | wgpu::بفر_Usage::CopySrc | wgpu::بفر_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_بفر_context * buf_ctx =
        جديد ggml_backend_webgpu_بفر_context(buf, buf_name, ctx->webgpu_global_ctx);

    أرجع ggml_backend_بفر_init(buft, ggml_backend_webgpu_بفر_interface, buf_ctx, size);
}

ساكن size_t ggml_backend_webgpu_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    أرجع dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
ساكن size_t ggml_backend_webgpu_بفر_type_get_max_اندازه(ggml_backend_بفر_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    أرجع dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه;
}

ساكن size_t ggml_backend_webgpu_بفر_type_get_alloc_اندازه(ggml_backend_بفر_type_t buft,
                                                             ثابت ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    بدل (tensor->op) {
        حالة GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            اكسر;
        حالة GGML_OP_سر_K:
            {
                ثابت ggml_tensor * src0 = tensor->src[0];
                إذا (src0) {
                    ثابت size_t ډک = حجم(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        ډک * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                ثابت ggml_tensor * Q            = tensor->src[0];
                ثابت ggml_tensor * K            = tensor->src[1];
                ثابت ggml_tensor * V            = tensor->src[2];
                ثابت ggml_tensor * ماسک         = tensor->src[3];
                ثابت خپلکاری &        capabilities = ctx->webgpu_global_ctx->capabilities;
                إذا (ggml_webgpu_flash_attn_use_vec_لار(ctx->webgpu_global_ctx, Q, K, V)) {
                    ثابت منطقي kv_مستقیم =
                        ggml_webgpu_flash_attn_k_مستقیم(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی) ||
                        ggml_webgpu_flash_attn_v_مستقیم(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی);
                    ثابت uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_اندازه, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        ماسک != تش_اشاره, kv_مستقیم);

                    ثابت uint32_t vec_nwg_cap = capabilities.min_subgroup_اندازه;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ثابت size_t   align = capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment;
                    ثابت uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    إذا (nwg > 1u) {
                        ثابت uint64_t tmp_معلومات_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ثابت uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ثابت size_t   tmp_اندازه_بایټونه = ROUNDUP_POW2((tmp_معلومات_elems + tmp_stats_elems) * حجم(عائم),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_اندازه_بایټونه + align;
                    } وإلا {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    إذا (ماسک != تش_اشاره) {
                        ثابت uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ثابت uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ثابت uint32_t stride_mask3    = (uint32_t) (ماسک->nb[3] / ggml_type_اندازه(ماسک->type));
                        ثابت uint32_t blk_batch_شمېر = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ثابت uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_شمېر;
                        ثابت size_t   blk_اندازه_بایټونه =
                            ROUNDUP_POW2(blk_elems * حجم(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_اندازه_بایټونه + align;
                    }
                    res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            اكسر;
        حالة GGML_OP_MUL_MAT:
            {
                ثابت ggml_tensor * src0 = tensor->src[0];
                ثابت ggml_tensor * src1 = tensor->src[1];
                منطقي                use_mmvq =
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_حاصل_ضرب,
                                             ctx->webgpu_global_ctx->vendor);
                إذا (use_mmvq) {
                    ثابت size_t q8_src1_اندازه = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_اندازه +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment,
                                       WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
            {
                ثابت ggml_tensor * src0 = tensor->src[0];
                ثابت ggml_tensor * src1 = tensor->src[1];
                إذا (src0 && src1) {
                    ثابت size_t ids_cache_اندازه = حجم(uint32_t) * tensor->ne[1] * tensor->ne[2];
                    ثابت size_t ids_tiles_اندازه = ggml_webgpu_mul_mat_id_tiles_nbytes(tensor);
                    res                         = ROUNDUP_POW2(
                        res + ids_cache_اندازه + ids_tiles_اندازه +
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment * 2,
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
                }
            }
            اكسر;
        افتراضي:
            اكسر;
    }
    أرجع res;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Device Interface */

ساكن ثابت حرف * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    أرجع ctx->device_name.c_str();
}

ساكن ثابت حرف * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    أرجع ctx->device_desc.c_str();
}

ساكن فراغ ggml_backend_webgpu_device_get_حافظه(ggml_backend_dev_t dev, size_t * حرر, size_t * ټول) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_بفر_اندازه = ctx->webgpu_global_ctx->capabilities.limits.max_بفر_اندازه_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#إذا UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_اندازه = static_cast<uint64_t>(UINTPTR_MAX);
    إذا (max_بفر_اندازه > max_ptr_اندازه) {
        max_بفر_اندازه = max_ptr_اندازه;
    }
#د_که_پای
    *حرر  = static_cast<size_t>(max_بفر_اندازه);
    *ټول = static_cast<size_t>(max_بفر_اندازه);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_ناکارول_شوی(dev);
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_ټول);
    props->caps = {
        /* .async                 = */ خطأ,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ خطأ,
        /* .mmap_support          = */ ریښتیا,
    };
}

ساكن ggml_guid_t ggml_backend_webgpu_guid(فراغ) {
    ساكن ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    أرجع &guid;
}

ساكن فراغ ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_تارونه = ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup *
                         ctx->capabilities.limits.max_Compute_Workgroups_فی_Dimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_بایټونه_فی_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه, max_تارونه);
    std::vector<wgpu::Constant_ثبت> constants(2);
    constants[0].کلی     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
    constants[1].کلی     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_بایټونه_فی_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

ساكن فراغ ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ثابت حرف * ثابت          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_شمېر = 2;
    options.بل_دننه_Chain                   = &adapterTogglesDesc;
#د_که_پای

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::Callback_ډېرپېښ::AllowSpontaneous,
            [&adapter](wgpu::Request_Adapter_حالت حالت, wgpu::Adapter _adapter, wgpu::StringView پیغام) {
                إذا (حالت != wgpu::Request_Adapter_حالت::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(پیغام).c_str());
                    أرجع;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

ساكن فراغ create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != تش_اشاره);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::Adapter_Properties_Subgroup_ماتریکس_Configs subgroup_ماتریکس_configs{};
    إذا (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ماتریکس)) {
        info.بل_دننه_Chain = &subgroup_ماتریکس_configs;
    }
#د_که_پای
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_اندازه = ggml_backend_webgpu_get_command_submit_batch_اندازه();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_حاصل_ضرب = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8_Integer_Dot_حاصل_ضرب);

    منطقي valid_subgroup_ماتریکس_config = خطأ;
#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    إذا (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ماتریکس)) {
        لكل (size_t i = 0; i < subgroup_ماتریکس_configs.config_شمېر; i++) {
            ثابت wgpu::Subgroup_ماتریکس_Config config = subgroup_ماتریکس_configs.configs[i];
            إذا (config.componentType == wgpu::Subgroup_ماتریکس_Component_Type::F16 &&
                config.پایله_Component_Type == wgpu::Subgroup_ماتریکس_Component_Type::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_ماتریکس_config                  = ریښتیا;
                اكسر;
            }
        }
    }
#د_که_پای
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_ماتریکس = valid_subgroup_ماتریکس_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_اندازه = info.subgroup_Min_اندازه;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_اندازه = info.subgroup_Max_اندازه;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    إذا (ctx->webgpu_global_ctx->capabilities.supports_subgroup_ماتریکس) {
        required_features.push_back(wgpu::FeatureName::Chromium_Experimental_Subgroup_ماتریکس);
    }
#د_که_پای

    إذا (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#د_که_پای

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.معلومات();
    dev_desc.required_Feature_شمېر = required_features.size();
    dev_desc.سېټ_Device_ورک_Callback(
        wgpu::Callback_ډېرپېښ::AllowSpontaneous,
        [](ثابت wgpu::Device & device, wgpu::Device_ورک_Reason reason, wgpu::StringView پیغام) {
            إذا (reason == wgpu::Device_ورک_Reason::Destroyed) {
                أرجع;
            }
            GGML_ناکارول_شوی(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<صحيح>(reason),
                           std::string(پیغام).c_str());
        });
    dev_desc.سېټ_Uncaptured_Error_Callback(
        [](ثابت wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView پیغام) {
            GGML_ناکارول_شوی(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<صحيح>(reason),
                       std::string(پیغام).c_str());
        });

#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Enable Dawn-specific toggles to increase native performance
    // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
    //       only for native performance?
    ثابت حرف * ثابت          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
                                                            "disable_polyfills_on_integer_div_and_mod" };
    ثابت حرف * ثابت          deviceDisabledToggles[] = { "timestamp_quantization" };
    wgpu::DawnTogglesDescriptor deviceTogglesDesc;
    deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
    deviceTogglesDesc.enabled_Toggle_شمېر  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_شمېر = 1;

    dev_desc.بل_دننه_Chain = &deviceTogglesDesc;
#د_که_پای

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::Callback_ډېرپېښ::AllowSpontaneous,
            [ctx](wgpu::Request_Device_حالت حالت, wgpu::Device device, wgpu::StringView پیغام) {
                إذا (حالت != wgpu::Request_Device_حالت::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(پیغام).c_str());
                    أرجع;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != تش_اشاره);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_بفر(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_اندازه_بایټونه, wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.Get_کتار();

    GGML_LOG_INFO(
        "ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | "
        "device_desc: %s\n",
        info.vendorID, std::string(info.vendor).c_str(), std::string(info.architecture).c_str(), info.deviceID,
        std::string(info.device).c_str(), std::string(info.description).c_str());
}

ساكن webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * dev_ctx    = (ggml_backend_webgpu_device_context *) dev->context;
    webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
    webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
    webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
    webgpu_ctx->param_arena.init(
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_اندازه_بایټونه,
        webgpu_ctx->global_ctx->command_submit_batch_اندازه + WEBGPU_عدد_PARAM_خونه_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_یوشان_بفر_دننه_واټن_Alignment);
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->سېټ_لیکې_dev_error_buf,
                              WEBGPU_سېټ_لیکې_ERROR_BUF_اندازه_بایټونه,
                              wgpu::بفر_Usage::Storage | wgpu::بفر_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->سېټ_لیکې_host_error_buf,
                              WEBGPU_سېټ_لیکې_ERROR_BUF_اندازه_بایټونه,
                              wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "set_rows_host_error_buf");

#که_تعریف_شوی GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = خطأ;
    ggml_webgpu_create_بفر(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_اندازه_بایټونه,
        wgpu::بفر_Usage::QueryResolve | wgpu::بفر_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_اندازه_بایټونه,
                              wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::Query_سېټ_Descriptor query_سېټ_desc = {};
    query_سېټ_desc.type                     = wgpu::QueryType::Timestamp;
    query_سېټ_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_شمېر;
    webgpu_ctx->profile_timestamp_query_سېټ = webgpu_ctx->global_ctx->device.Create_Query_سېټ(&query_سېټ_desc);
#د_که_پای

#که_تعریف_شوی GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * حجم(uint32_t),
                              wgpu::بفر_Usage::CopyDst | wgpu::بفر_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_بفر(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * حجم(uint32_t),
                              wgpu::بفر_Usage::Storage | wgpu::بفر_Usage::CopySrc, "debug_dev_buf");
#د_که_پای
    أرجع webgpu_ctx;
}

ساكن ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_ناکارول_شوی(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    خپلکاری * backend_ctx      = جديد ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    خپلکاری * backend = جديد ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    أرجع backend;
}

ساكن ggml_backend_بفر_type_t ggml_backend_webgpu_device_get_بفر_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    ساكن بنية ggml_backend_بفر_type ggml_backend_webgpu_بفر_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_بفر_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_بفر_type_alloc_بفر,
                        /* .alloc_buffer_n      = */ عدم,
                        /* .get_alignment       = */ ggml_backend_webgpu_بفر_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_بفر_type_get_max_اندازه,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_بفر_type_get_alloc_اندازه,
                        /* .get_alloc_size_n    = */ عدم,
                        /* .is_host             = */ عدم,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ عدم
    };

    أرجع &ggml_backend_webgpu_بفر_type;
}

ساكن منطقي ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    GGML_ناکارول_شوی(dev);
    أرجع buft->iface.get_name == ggml_backend_webgpu_بفر_type_get_name;
}

ساكن منطقي ggml_webgpu_supported_qtype(ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
            أرجع ریښتیا;
        افتراضي:
            أرجع خطأ;
    }
}

ساكن منطقي ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * src2 = op->src[2];

    // on smaller devices (or CI), tensors may be larger than the max storage buffer size
    إذا (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه ||
        (src0 != تش_اشاره &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه) ||
        (src1 != تش_اشاره &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه)) {
        أرجع خطأ;
    }

    منطقي supports_op = خطأ;
    بدل (op->op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلول:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE:
            supports_op = ریښتیا;
            اكسر;
        حالة GGML_OP_جمع_کړه:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            اكسر;
        حالة GGML_OP_جمع_کړه_ID:
            supports_op = src0->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_CONCAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            اكسر;
        حالة GGML_OP_REPEAT:
            supports_op = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_I32 ||
                           src0->type == GGML_TYPE_I16);
            اكسر;
        حالة GGML_OP_DUP:
        حالة GGML_OP_CPY:
        حالة GGML_OP_CONT:
            supports_op = (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                          (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
            اكسر;
        حالة GGML_OP_سېټ:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            اكسر;
        حالة GGML_OP_سېټ_لیکې:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            اكسر;
        حالة GGML_OP_GET_لیکې:
            إذا (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                ggml_webgpu_supported_qtype(src0->type)) {
                supports_op = (op->type == GGML_TYPE_F32);
            } وإلا إذا (src0->type == GGML_TYPE_I32) {
                supports_op = op->type == GGML_TYPE_I32;
            }
            اكسر;
        حالة GGML_OP_MUL_MAT:
            {
                بدل (src1->type) {
                    حالة GGML_TYPE_F16:
                        supports_op |= (src0->type == GGML_TYPE_F16);
                        اكسر;
                    حالة GGML_TYPE_F32:
                        بدل (src0->type) {
                            حالة GGML_TYPE_F32:
                            حالة GGML_TYPE_F16:
                            حالة GGML_TYPE_BF16:
                            حالة GGML_TYPE_Q1_0:
                            حالة GGML_TYPE_Q2_0:
                            حالة GGML_TYPE_Q4_0:
                            حالة GGML_TYPE_Q4_1:
                            حالة GGML_TYPE_Q5_0:
                            حالة GGML_TYPE_Q5_1:
                            حالة GGML_TYPE_Q8_0:
                            حالة GGML_TYPE_Q2_K:
                            حالة GGML_TYPE_Q3_K:
                            حالة GGML_TYPE_Q4_K:
                            حالة GGML_TYPE_Q5_K:
                            حالة GGML_TYPE_Q6_K:
                            حالة GGML_TYPE_IQ2_XXS:
                            حالة GGML_TYPE_IQ2_XS:
                            حالة GGML_TYPE_IQ2_S:
                            حالة GGML_TYPE_IQ3_XXS:
                            حالة GGML_TYPE_IQ3_S:
                            حالة GGML_TYPE_IQ1_S:
                            حالة GGML_TYPE_IQ1_M:
                            حالة GGML_TYPE_IQ4_NL:
                            حالة GGML_TYPE_IQ4_XS:
                            حالة GGML_TYPE_MXFP4:
                            حالة GGML_TYPE_NVFP4:
                                supports_op = ریښتیا;
                                اكسر;
                            افتراضي:
                                اكسر;
                        }
                    افتراضي:
                        اكسر;
                }
                اكسر;
            }
        حالة GGML_OP_MUL_MAT_ID:
            بدل (src1->type) {
                حالة GGML_TYPE_F16:
                    supports_op |= (src0->type == GGML_TYPE_F16);
                    اكسر;
                حالة GGML_TYPE_F32:
                    بدل (src0->type) {
                        حالة GGML_TYPE_F32:
                        حالة GGML_TYPE_F16:
                        حالة GGML_TYPE_BF16:
                        حالة GGML_TYPE_Q1_0:
                        حالة GGML_TYPE_Q4_0:
                        حالة GGML_TYPE_Q4_1:
                        حالة GGML_TYPE_Q5_0:
                        حالة GGML_TYPE_Q5_1:
                        حالة GGML_TYPE_Q8_0:
                        حالة GGML_TYPE_Q2_K:
                        حالة GGML_TYPE_Q3_K:
                        حالة GGML_TYPE_Q4_K:
                        حالة GGML_TYPE_Q5_K:
                        حالة GGML_TYPE_Q6_K:
                        حالة GGML_TYPE_IQ1_S:
                        حالة GGML_TYPE_IQ1_M:
                        حالة GGML_TYPE_IQ2_XXS:
                        حالة GGML_TYPE_IQ2_XS:
                        حالة GGML_TYPE_IQ2_S:
                        حالة GGML_TYPE_IQ3_XXS:
                        حالة GGML_TYPE_IQ3_S:
                        حالة GGML_TYPE_IQ4_NL:
                        حالة GGML_TYPE_IQ4_XS:
                        حالة GGML_TYPE_MXFP4:
                        حالة GGML_TYPE_NVFP4:
                            supports_op = ریښتیا;
                            اكسر;
                        افتراضي:
                            اكسر;
                    }
                    اكسر;
                افتراضي:
                    اكسر;
            }
            إذا (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                supports_op = خطأ;
            }
            اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                // conservative support checks for whether the more resource-intensive shader paths
                // can be used, to avoid cases where flash_attn is assigned to the CPU later on
                supports_op = src0->type == GGML_TYPE_F32 &&
                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
                              (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16 ||
                               src2->type == GGML_TYPE_Q4_0 || src2->type == GGML_TYPE_Q8_0) &&
                              op->type == GGML_TYPE_F32;
                إذا (!supports_op) {
                    اكسر;
                }
                إذا (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                    supports_op = خطأ;
                    اكسر;
                }
                ثابت خپلکاری & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ثابت size_t storage_دننه_واټن_alignment = capabilities.limits.min_Storage_بفر_دننه_واټن_Alignment;

                // subgroup matrix path requirements
                ثابت منطقي use_subgroup_ماتریکس = ggml_webgpu_flash_attn_can_use_subgroup_ماتریکس_لار(
                    capabilities.supports_subgroup_ماتریکس, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ثابت منطقي float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_دننه_واټن_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_دننه_واټن_alignment));
                ثابت uint32_t k_tile_سر_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی :
                                                       (uint32_t) ggml_blck_اندازه(src1->type);
                ثابت uint32_t v_tile_سر_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_پلنوالی :
                                                       (uint32_t) ggml_blck_اندازه(src2->type);
                ثابت منطقي     tile_kv_سر_dims_aligned =
                    src0->ne[0] % k_tile_سر_align == 0 && src2->ne[0] % v_tile_سر_align == 0;
                ثابت منطقي tile_can_dispatch_all_q_لیکې =
                    capabilities.limits.max_Compute_Invocations_فی_Workgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_اندازه;
                ثابت منطقي use_tile = !use_subgroup_ماتریکس && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_سر_dims_aligned && tile_can_dispatch_all_q_لیکې;

                إذا (!use_subgroup_ماتریکس && !use_tile) {
                    supports_op = خطأ;
                    اكسر;
                }
                ثابت uint32_t q_tile =
                    use_subgroup_ماتریکس ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ثابت uint32_t kv_granularity = use_subgroup_ماتریکس ? capabilities.sg_mat_n : 1u;
                ثابت منطقي     kv_مستقیم = use_subgroup_ماتریکس ?
                                               ggml_webgpu_flash_attn_k_مستقیم(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_مستقیم(src0, src2, capabilities.sg_mat_k) :
                                               خطأ;
                ثابت uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_اندازه, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != تش_اشاره, kv_مستقیم);
                supports_op = max_kv_tile > 0;
                اكسر;
            }
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_لیکې(src0);
            اكسر;
        حالة GGML_OP_ROPE:
            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
            اكسر;
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(op)) {
                حالة GGML_GLU_OP_REGLU:
                حالة GGML_GLU_OP_GEGLU:
                حالة GGML_GLU_OP_SWIGLU:
                حالة GGML_GLU_OP_GEGLU_ERF:
                حالة GGML_GLU_OP_GEGLU_چټک:
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU_OAI:
                    supports_op = op->type == GGML_TYPE_F32;
                    اكسر;
                افتراضي:
                    اكسر;
            }
            اكسر;
        حالة GGML_OP_SCALE:
            supports_op = op->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            supports_op = op->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_UNARY:
            {
                ثابت ggml_unary_op UNARY_OP = ggml_get_unary_op(op);

                بدل (UNARY_OP) {
                    حالة GGML_UNARY_OP_ABS:
                    حالة GGML_UNARY_OP_SGN:
                    حالة GGML_UNARY_OP_NEG:
                    حالة GGML_UNARY_OP_ګام:
                    حالة GGML_UNARY_OP_TANH:
                    حالة GGML_UNARY_OP_ELU:
                    حالة GGML_UNARY_OP_RELU:
                    حالة GGML_UNARY_OP_SIGMOID:
                    حالة GGML_UNARY_OP_GELU:
                    حالة GGML_UNARY_OP_GELU_چټک:
                    حالة GGML_UNARY_OP_SILU:
                    حالة GGML_UNARY_OP_HARDSWISH:
                    حالة GGML_UNARY_OP_HARDSIGMOID:
                    حالة GGML_UNARY_OP_EXP:
                    حالة GGML_UNARY_OP_GELU_ERF:
                    حالة GGML_UNARY_OP_SOFTPLUS:
                    حالة GGML_UNARY_OP_EXPM1:
                    حالة GGML_UNARY_OP_FLOOR:
                    حالة GGML_UNARY_OP_CEIL:
                    حالة GGML_UNARY_OP_پړاو:
                    حالة GGML_UNARY_OP_TRUNC:
                    حالة GGML_UNARY_OP_XIELU:
                        supports_op =
                            (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
                        اكسر;
                    افتراضي:
                        اكسر;
                }
            }
            اكسر;
        حالة GGML_OP_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_DIAG:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_SOLVE_TRI:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_CONV_2D:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
            اكسر;
        حالة GGML_OP_CONV_2D_DW:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                          (ggml_is_contiguous(src1) || ggml_is_contiguous_channels(src1));
            اكسر;
        حالة GGML_OP_IM2_ستون:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            اكسر;
        حالة GGML_OP_SSM_CONV:
            supports_op = op->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_SSM_لټون:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
            اكسر;
        حالة GGML_OP_GATED_DELTA_NET:
            {
                ثابت uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_فی_Workgroup;
            }
            اكسر;
        حالة GGML_OP_CLAMP:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            اكسر;
        حالة GGML_OP_FILL:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            اكسر;
        حالة GGML_OP_LOG:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            اكسر;
        حالة GGML_OP_SQR:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            اكسر;
        حالة GGML_OP_SQRT:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            اكسر;
        حالة GGML_OP_SIN:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            اكسر;
        حالة GGML_OP_COS:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type);
            اكسر;
        حالة GGML_OP_PAD:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_ARGMAX:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32;
            اكسر;
        حالة GGML_OP_ARGSORT:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_لیکې(src0);
            اكسر;
        حالة GGML_OP_سر_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_لیکې(src0);
            اكسر;
        حالة GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            اكسر;
        حالة GGML_OP_مجموعه:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            اكسر;
        حالة GGML_OP_مجموعه_لیکې:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_لیکې(src0);
            اكسر;
        حالة GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            اكسر;
        افتراضي:
            اكسر;
    }
    إذا (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه ||
        (src0 != تش_اشاره &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه) ||
        (src1 != تش_اشاره &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه) ||
        (src2 != تش_اشاره &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_بفر_Binding_اندازه)) {
        supports_op = خطأ;
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported due to size: ");
    }

    إذا (!supports_op) {
        WEBGPU_LOG_DEBUG("ggml_webgpu op not supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    } وإلا {
        WEBGPU_LOG_DEBUG("ggml_webgpu op supported: "
                         << ggml_op_name(op->op) << " with types dst: " << ggml_type_name(op->type)
                         << ", src0: " << (op->src[0] ? ggml_type_name(op->src[0]->type) : "null")
                         << ", src1: " << (op->src[1] ? ggml_type_name(op->src[1]->type) : "null"));
    }
    أرجع supports_op;
}

ساكن بنية ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_حافظه,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_backend_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_بفر_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ عدم,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ عدم,
    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

ساكن ثابت حرف * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    أرجع ctx->name;
}

ساكن size_t ggml_backend_webgpu_reg_get_device_شمېر(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    أرجع ctx->device_شمېر;
}

// Only one device is supported for now
ساكن ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t شاخص) {
    GGML_ASSERT(شاخص == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_ټول_پیل(reg_get_device);

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    create_webgpu_device(reg_ctx);

    ساكن ggml_backend_webgpu_device_context device_ctx;
    device_ctx.device_name            = GGML_WEBGPU_NAME;
    device_ctx.device_desc            = GGML_WEBGPU_NAME;
    device_ctx.webgpu_global_ctx      = reg_ctx->webgpu_global_ctx;
    // See GGML Backend Device Interface section
    ساكن ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };

    WEBGPU_CPU_PROFILE_ټول_پای_ټکی(reg_get_device, reg_ctx->webgpu_global_ctx);
    أرجع &device;
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_شمېر,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ عدم,
};

/* End GGML Backend Registration Interface */

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    // Intentionally leak the global registry context to avoid crashing inside
    // Dawn/Vulkan static teardown during process exit.
    ساكن ggml_backend_webgpu_reg_context * ctx = جديد ggml_backend_webgpu_reg_context();

    ساكن ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخه,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ ctx,
    };

    ctx->name         = GGML_WEBGPU_NAME;
    ctx->device_شمېر = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    إذا (ctx->webgpu_global_ctx != تش_اشاره && ctx->webgpu_global_ctx->instance != تش_اشاره) {
        أرجع &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.معلومات();
    instance_descriptor.required_Feature_شمېر                 = instance_features.size();

#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ثابت حرف * ثابت          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_شمېر = 1;
    instance_descriptor.بل_دننه_Chain        = &instanceTogglesDesc;
#د_که_پای

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(جديد webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    إذا (ctx->webgpu_global_ctx->instance != تش_اشاره) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    إذا (adapter != تش_اشاره && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#د_که_پای
    ) {
        ctx->device_شمېر = 1;
    }

    أرجع &reg;
}

ggml_backend_t ggml_backend_webgpu_init(فراغ) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    إذا (ggml_backend_reg_dev_شمېر(reg) == 0) {
        أرجع تش_اشاره;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    أرجع ggml_backend_webgpu_backend_init(dev, تش_اشاره);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
