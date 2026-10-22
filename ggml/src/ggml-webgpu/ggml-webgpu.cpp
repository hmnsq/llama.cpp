/*
    WebGPU backend implementation.
    Note: Use ClangFormat to format this file.
*/

#ضمن "ggml-webgpu.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-webgpu-shader-lib.hpp"
#ضمن "ggml.h"

#اگر_تعریف_شده __EMSCRIPTEN__
#    include <emscripten/emscripten.h>
#پایان_اگر

#ضمن <webgpu/webgpu_cpp.h>

#ضمن <atomic>
#ضمن <cstdint>
#ضمن <cstring>
#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
#    include <iomanip>
#پایان_اگر
#إذا defined(GGML_WEBGPU_DEBUG) || defined(GGML_WEBGPU_CPU_PROFILE) || defined(GGML_WEBGPU_GPU_PROFILE)
#    include <iostream>
#پایان_اگر
#ضمن <memory>
#ضمن <mutex>
#ضمن <optional>
#ضمن <string>
#ضمن <utility>
#ضمن <vector>

#تعریف_کن ROUNDUP_POW2(x, pow2) (((x) + ((pow2) - 1)) & ~((pow2) - 1))
#تعریف_کن CEIL_DIV(M, N)        (((M) + (N) - 1) / (N))

// Return a rectangular grid of workgroups with minimal over-provisioned workgroups.
// Assumes that the total number of workgroups does not exceed max_per_dim^2.
ساكن مضمن فراغ compute_2d_workgroups(uint32_t مجموع_wg, uint32_t max_در_هر_dim, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(مجموع_wg, max_در_هر_dim));
    wg_x = CEIL_DIV(مجموع_wg, wg_y);
}

ساكن مضمن uint32_t ggml_webgpu_u32_from_f32(عائم value) {
    uint32_t بیتها;
    کپی_حافظه(&بیتها, &value, حجم(بیتها));
    أرجع بیتها;
}

#اگر_تعریف_شده GGML_WEBGPU_DEBUG
#    define WEBGPU_LOG_DEBUG(پیام_2)  std::خروجی << پیام_2 << std::پایان_سطر
#    define WEBGPU_DEBUG_BUF_ELEMS 512
#وإلا
#    define WEBGPU_LOG_DEBUG(پیام_2) ((فراغ) 0)
#پایان_اگر  // GGML_WEBGPU_DEBUG

#اگر_تعریف_شده GGML_WEBGPU_CPU_PROFILE
// total timing (aggregated)
#    define WEBGPU_CPU_PROFILE_مجموع_شروع(id) خودکار cpu_مجموع_شروع##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_مجموع_انتها(id, ctx)                                                         \
        خودکار   cpu_مجموع_انتها##id = std::chrono::high_resolution_clock::now();                            \
        مزدوج cpu_مجموع_زمان##id =                                                                      \
            std::chrono::duration<مزدوج, std::milli>(cpu_مجموع_انتها##id - cpu_مجموع_شروع##id).count(); \
        (ctx)->cpu_زمان_ms[#id] += cpu_مجموع_زمان##id;
// fine-grained timing (not included in totals)
#    define WEBGPU_CPU_PROFILE_DETAIL_شروع(id) خودکار cpu_detail_شروع##id = std::chrono::high_resolution_clock::now();

#    define WEBGPU_CPU_PROFILE_DETAIL_انتها(id, ctx)                                                          \
        خودکار   cpu_detail_انتها##id = std::chrono::high_resolution_clock::now();                             \
        مزدوج cpu_detail_زمان##id =                                                                       \
            std::chrono::duration<مزدوج, std::milli>(cpu_detail_انتها##id - cpu_detail_شروع##id).count(); \
        (ctx)->cpu_detail_ms[#id] += cpu_detail_زمان##id;
#وإلا
#    define WEBGPU_CPU_PROFILE_مجموع_شروع(id)
#    define WEBGPU_CPU_PROFILE_مجموع_انتها(id, ctx)
#    define WEBGPU_CPU_PROFILE_DETAIL_شروع(id)
#    define WEBGPU_CPU_PROFILE_DETAIL_انتها(id, ctx)
#پایان_اگر  // GGML_WEBGPU_CPU_PROFILE

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
#    define WEBGPU_MAX_PROFILE_QUERY_شمار        4096u
#    define WEBGPU_TIMESTAMP_QUERY_BUF_اندازه_بایتها (WEBGPU_MAX_PROFILE_QUERY_شمار * حجم(uint64_t))
#پایان_اگر

/* Constants */

#تعریف_کن WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_اندازه 64u
#تعریف_کن WEBGPU_رقم_PARAM_جایگاه_SAFETY_MARGIN      10u
#تعریف_کن WEBGPU_RUNTIME_WAIT_TIMEOUT_MS           30000u
#تعریف_کن WEBGPU_RUNTIME_WAIT_TIMEOUT_NS           (WEBGPU_RUNTIME_WAIT_TIMEOUT_MS * 1e6)
#تعریف_کن WEBGPU_PARAMS_BUF_اندازه_بایتها             128  // enough for 32 parameters
#تعریف_کن WEBGPU_مجموعه_ردیفها_ERROR_BUF_اندازه_بایتها     4
#تعریف_کن WEBGPU_STORAGE_BUF_BINDING_MULT          4    // a storage buffer binding size must be a multiple of 4

/* End Constants */

// This is a "fake" base pointer, since WebGPU buffers do not have pointers to
// their locations.
ساكن فراغ * ثابت webgpu_ptr_پایه = (فراغ *) (uintptr_t) 0x1000;  // NOLINT

ساكن size_t ggml_webgpu_tensor_جابه_جایی(ثابت ggml_tensor * tensor) {
    ثابت ggml_tensor * پایه_tensor = tensor->view_src ? tensor->view_src : tensor;
    أرجع (size_t) ((uintptr_t) پایه_tensor->داده - (uintptr_t) webgpu_ptr_پایه) + tensor->view_offs;
}

/* Struct definitions */

// Forward reference
ساكن فراغ ggml_webgpu_create_میانگیر(wgpu::Device &    device,
                                      wgpu::میانگیر_2 &    میانگیر,
                                      size_t            size,
                                      wgpu::میانگیر_Usage usage,
                                      ثابت حرف *      label);

// Slot-based parameter arena for compute graph encoding. Each encoded kernel
// gets a unique uniform-buffer slice within the current batch, and the slot
// cursor is reset immediately after that batch is submitted.
بنية webgpu_param_arena {
    wgpu::میانگیر_2 میانگیر;
    size_t       جایگاه_stride = 0;
    size_t       جایگاه_اندازه   = 0;
    uint32_t     جایگاه_شمار  = 0;
    uint32_t     بعدی_جایگاه   = 0;

    فراغ init(wgpu::Device device, size_t جایگاه_اندازه, uint32_t جایگاه_شمار, size_t alignment) {
        هذا->جایگاه_stride = ROUNDUP_POW2(جایگاه_اندازه, alignment);
        هذا->جایگاه_اندازه   = جایگاه_اندازه;
        هذا->جایگاه_شمار  = جایگاه_شمار;
        هذا->بعدی_جایگاه   = 0;

        ggml_webgpu_create_میانگیر(device, میانگیر, هذا->جایگاه_stride * جایگاه_شمار,
                                  wgpu::میانگیر_Usage::CopyDst | wgpu::میانگیر_Usage::Uniform, "ggml_webgpu_param_arena");
    }

    size_t alloc_جایگاه(size_t size) {
        GGML_ASSERT(size <= جایگاه_اندازه);
        إذا (بعدی_جایگاه >= جایگاه_شمار) {
            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
        }

        أرجع جایگاه_stride * بعدی_جایگاه++;
    }

    فراغ reset() { بعدی_جایگاه = 0; }

    فراغ cleanup() {
        إذا (میانگیر) {
            میانگیر.Destroy();
            میانگیر = اشاره_تهی;
        }
    }

    ~webgpu_param_arena() { هذا->cleanup(); }
};

بنية webgpu_encoded_op {
    uint32_t رقم_kernels = 0;
#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    std::vector<std::string> pipeline_نامها;
#پایان_اگر
};

بنية webgpu_dispatch_desc {
    webgpu_pipeline                   pipeline;
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_گروه_مدخل> bind_گروه_entries;
    std::pair<uint32_t, uint32_t>     workgroups = { 1, 1 };
};

بنية webgpu_capabilities {
    wgpu::Limits limits;
    منطقي         supports_subgroups       = خطأ;
    منطقي         supports_subgroup_ماتریس = خطأ;
    منطقي         supports_dot_حاصلضرب     = خطأ;

    uint32_t sg_mat_m = 0;
    uint32_t sg_mat_n = 0;
    uint32_t sg_mat_k = 0;

    uint32_t subgroup_اندازه     = 0;
    uint32_t min_subgroup_اندازه = 0;
    uint32_t max_subgroup_اندازه = 0;
    size_t   memset_بایتها_در_هر_thread;
};

// Stores global webgpu members
بنية webgpu_global_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter  adapter;
    wgpu::Device   device;
    wgpu::صف_2    queue;
    uint32_t       command_submit_batch_اندازه = WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_اندازه;
    uint32_t       max_inflight_batches      = UINT32_MAX;

    webgpu_capabilities  capabilities;
    // Shared buffer to move data from device to host
    wgpu::میانگیر_2         get_tensor_staging_buf;
    // Global mutex for get_tensor
    std::recursive_mutex mutex;

    wgpu::میانگیر_2    memset_params_buf;
    webgpu_pipeline memset_pipeline;

    std::string vendor;

    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
#اگر_تعریف_شده GGML_WEBGPU_CPU_PROFILE
    // Profiling: labeled CPU time in ms (total)
    std::unordered_map<std::string, مزدوج> cpu_زمان_ms;
    // Profiling: detailed CPU time in ms
    std::unordered_map<std::string, مزدوج> cpu_detail_ms;
#پایان_اگر

#اگر_تعریف_شده GGML_WEBGPU_DEBUG
    wgpu::میانگیر_2 debug_host_buf;
    wgpu::میانگیر_2 debug_dev_buf;
#پایان_اگر

    ~webgpu_global_context_struct() {
        إذا (هذا->get_tensor_staging_buf) {
            هذا->get_tensor_staging_buf.Destroy();
            هذا->get_tensor_staging_buf = اشاره_تهی;
        }
        إذا (هذا->memset_params_buf) {
            هذا->memset_params_buf.Destroy();
            هذا->memset_params_buf = اشاره_تهی;
        }
#اگر_تعریف_شده GGML_WEBGPU_DEBUG
        إذا (هذا->debug_host_buf) {
            هذا->debug_host_buf.Destroy();
            هذا->debug_host_buf = اشاره_تهی;
        }
        إذا (هذا->debug_dev_buf) {
            هذا->debug_dev_buf.Destroy();
            هذا->debug_dev_buf = اشاره_تهی;
        }
#پایان_اگر
    }
};

تعريفنوع std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;

// All the base objects needed to run operations on a WebGPU device
بنية webgpu_context_struct {
    // Points to global instances owned by ggml_backend_webgpu_reg_context
    webgpu_global_context global_ctx;

    std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;

    webgpu_param_arena       param_arena;
    wgpu::میانگیر_2             مجموعه_ردیفها_dev_error_buf;
    wgpu::میانگیر_2             مجموعه_ردیفها_host_error_buf;
    wgpu::CommandEncoder     فعال_command_encoder;
    wgpu::ComputePassEncoder فعال_compute_pass;
    منطقي                     batch_compute_passes = درست;

    size_t memset_بایتها_در_هر_thread;

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    // Profiling: per-shader GPU time in ms
    std::unordered_map<std::string, مزدوج> shader_gpu_زمان_ms;
    wgpu::میانگیر_2                            profile_timestamp_dev_buf;
    wgpu::میانگیر_2                            profile_timestamp_host_buf;
    wgpu::Query_مجموعه                          profile_timestamp_query_مجموعه;
    uint32_t                                profile_timestamp_query_شمار = 0;
#پایان_اگر

    ~webgpu_context_struct() {
#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
        إذا (هذا->profile_timestamp_host_buf) {
            هذا->profile_timestamp_host_buf.Destroy();
            هذا->profile_timestamp_host_buf = اشاره_تهی;
        }
        إذا (هذا->profile_timestamp_dev_buf) {
            هذا->profile_timestamp_dev_buf.Destroy();
            هذا->profile_timestamp_dev_buf = اشاره_تهی;
        }
        إذا (هذا->profile_timestamp_query_مجموعه) {
            هذا->profile_timestamp_query_مجموعه.Destroy();
            هذا->profile_timestamp_query_مجموعه = اشاره_تهی;
        }
#پایان_اگر
        إذا (هذا->مجموعه_ردیفها_host_error_buf) {
            هذا->مجموعه_ردیفها_host_error_buf.Destroy();
            هذا->مجموعه_ردیفها_host_error_buf = اشاره_تهی;
        }
        إذا (هذا->مجموعه_ردیفها_dev_error_buf) {
            هذا->مجموعه_ردیفها_dev_error_buf.Destroy();
            هذا->مجموعه_ردیفها_dev_error_buf = اشاره_تهی;
        }
    }
};

تعريفنوع std::shared_ptr<webgpu_context_struct> webgpu_context;

// Metadata required for the ggml backend registration/discovery interface
بنية ggml_backend_webgpu_reg_context {
    // Since the Instance is a global entrypoint into the WebGPU API, it lives here
    webgpu_global_context webgpu_global_ctx;
    size_t                device_شمار;
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
بنية ggml_backend_webgpu_میانگیر_context {
    wgpu::میانگیر_2          میانگیر;
    std::string           label;
    webgpu_global_context global_ctx;

    ggml_backend_webgpu_میانگیر_context(wgpu::میانگیر_2 buf, std::string lbl, webgpu_global_context global_ctx_) :
        میانگیر(std::move(buf)),
        label(std::move(lbl)),
        global_ctx(std::move(global_ctx_)) {}
};

/* WebGPU object initializations */

ساكن webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device &                           device,
                                                   ثابت حرف *                             shader_کد,
                                                   ثابت حرف *                             label,
                                                   ثابت std::vector<wgpu::Constant_مدخل> & constants = {}) {
    wgpu::Shader_منبع_WGSL shader_منبع;
    shader_منبع.code = shader_کد;

    wgpu::ShaderModuleDescriptor shader_desc;
    shader_desc.بعدی_درون_Chain = &shader_منبع;

    wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);

    wgpu::ComputePipelineDescriptor pipeline_desc;
    pipeline_desc.label              = label;
    pipeline_desc.compute.module     = shader_module;
    pipeline_desc.compute.مدخل_نقطه_2 = "main";   // Entry point in the WGSL code
    pipeline_desc.layout             = اشاره_تهی;  // nullptr means auto layout
    إذا (constants.size() > 0) {
        pipeline_desc.compute.constants     = constants.داده();
        pipeline_desc.compute.constant_شمار = constants.size();
    }
    أرجع { device.CreateComputePipeline(&pipeline_desc), label };
}

ساكن فراغ ggml_webgpu_create_میانگیر(wgpu::Device &    device,
                                      wgpu::میانگیر_2 &    میانگیر,
                                      size_t            size,
                                      wgpu::میانگیر_Usage usage,
                                      ثابت حرف *      label) {
    wgpu::میانگیر_Descriptor میانگیر_desc;
    میانگیر_desc.size             = size;
    میانگیر_desc.usage            = usage;
    میانگیر_desc.label            = label;
    میانگیر_desc.mappedAtCreation = خطأ;

    // TODO: error handling
    میانگیر = device.Create_میانگیر(&میانگیر_desc);
}

ساكن wgpu::میانگیر_2 ggml_webgpu_tensor_buf(ثابت ggml_tensor * tensor) {
    ggml_backend_webgpu_میانگیر_context * ctx = (ggml_backend_webgpu_میانگیر_context *) tensor->میانگیر->context;
    أرجع ctx->میانگیر;
}

// Binding offset for a tensor: the largest aligned offset at or before the tensor whose
// distance to the tensor is a whole number of type blocks, so shaders can index the
// misalignment in elements even for block quantized types.
ساكن size_t ggml_webgpu_tensor_align_جابه_جایی(ثابت ggml_tensor * t, size_t alignment) {
    ثابت size_t جابه_جایی    = ggml_webgpu_tensor_جابه_جایی(t);
    ثابت size_t type_اندازه = ggml_type_اندازه(t->type);
    size_t       aligned   = جابه_جایی & ~(alignment - 1);
    طالما ((جابه_جایی - aligned) % type_اندازه != 0) {
        GGML_ASSERT(aligned >= alignment);
        aligned -= alignment;
    }
    أرجع aligned;
}

ساكن size_t ggml_webgpu_tensor_misalignment(ثابت ggml_tensor * t, size_t alignment) {
    أرجع ggml_webgpu_tensor_جابه_جایی(t) - ggml_webgpu_tensor_align_جابه_جایی(t, alignment);
}

ساكن size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, ثابت ggml_tensor * t) {
    أرجع ggml_webgpu_tensor_misalignment(t, ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
}

ساكن size_t ggml_webgpu_tensor_align_جابه_جایی(webgpu_context & ctx, ثابت ggml_tensor * t) {
    أرجع ggml_webgpu_tensor_align_جابه_جایی(t, ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
}

ساكن size_t ggml_webgpu_tensor_binding_اندازه(ثابت ggml_tensor * t, size_t alignment) {
    أرجع ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(t, alignment),
                        WEBGPU_STORAGE_BUF_BINDING_MULT);
}

ساكن size_t ggml_webgpu_tensor_binding_اندازه(webgpu_context & ctx, ثابت ggml_tensor * t) {
    أرجع ggml_webgpu_tensor_binding_اندازه(t, ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
}

ساكن منطقي ggml_webgpu_tensor_binding_overlap(ثابت webgpu_global_context & global_ctx,
                                               ثابت ggml_tensor *           a,
                                               ثابت ggml_tensor *           b) {
    إذا (a->میانگیر != b->میانگیر) {
        أرجع خطأ;
    }

    ثابت size_t alignment = global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment;
    ثابت size_t a_جابه_جایی  = ggml_webgpu_tensor_align_جابه_جایی(a, alignment);
    ثابت size_t b_جابه_جایی  = ggml_webgpu_tensor_align_جابه_جایی(b, alignment);
    أرجع a_جابه_جایی < b_جابه_جایی + ggml_webgpu_tensor_binding_اندازه(b, alignment) &&
           b_جابه_جایی < a_جابه_جایی + ggml_webgpu_tensor_binding_اندازه(a, alignment);
}

ساكن منطقي ggml_webgpu_tensor_binding_overlap_range(ثابت webgpu_global_context & global_ctx,
                                                     ggml_tensor *                 tensor,
                                                     ggml_backend_میانگیر_t         میانگیر,
                                                     size_t                        جابه_جایی,
                                                     size_t                        size) {
    إذا (tensor->میانگیر != میانگیر) {
        أرجع خطأ;
    }

    ثابت size_t alignment     = global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment;
    ثابت size_t tensor_جابه_جایی = ggml_webgpu_tensor_align_جابه_جایی(tensor, alignment);
    أرجع tensor_جابه_جایی < جابه_جایی + size && جابه_جایی < tensor_جابه_جایی + ggml_webgpu_tensor_binding_اندازه(tensor, alignment);
}

بنية ggml_webgpu_merged_binding_range {
    size_t جابه_جایی;
    size_t size;
};

ساكن ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
    webgpu_context &                     ctx,
    std::initializer_list<ggml_tensor *> tensors) {
    size_t merged_جابه_جایی = SIZE_MAX;
    size_t merged_انتها    = 0;

    لكل (ggml_tensor * tensor : tensors) {
        ثابت size_t bind_جابه_جایی = ggml_webgpu_tensor_align_جابه_جایی(ctx, tensor);
        ثابت size_t bind_انتها    = bind_جابه_جایی + ggml_webgpu_tensor_binding_اندازه(ctx, tensor);

        merged_جابه_جایی = std::min(merged_جابه_جایی, bind_جابه_جایی);
        merged_انتها    = std::max(merged_انتها, bind_انتها);
    }

    أرجع { merged_جابه_جایی, merged_انتها - merged_جابه_جایی };
}

ساكن uint32_t ggml_webgpu_tensor_merged_element_جابه_جایی(ثابت ggml_tensor *                      tensor,
                                                         ثابت ggml_webgpu_merged_binding_range & merged_range) {
    أرجع (uint32_t) ((ggml_webgpu_tensor_جابه_جایی(tensor) - merged_range.جابه_جایی) / ggml_type_اندازه(tensor->type));
}

ساكن wgpu::Bind_گروه_مدخل ggml_webgpu_make_bind_گروه_مدخل(uint32_t     binding,
                                                              wgpu::میانگیر_2 میانگیر,
                                                              uint64_t     جابه_جایی,
                                                              uint64_t     size) {
    wgpu::Bind_گروه_مدخل مدخل = {};
    مدخل.binding              = binding;
    مدخل.میانگیر               = std::move(میانگیر);
    مدخل.جابه_جایی               = جابه_جایی;
    مدخل.size                 = size;
    أرجع مدخل;
}

ساكن wgpu::Bind_گروه_مدخل ggml_webgpu_make_tensor_bind_گروه_مدخل(webgpu_context & ctx,
                                                                     uint32_t         binding,
                                                                     ggml_tensor *    tensor) {
    أرجع ggml_webgpu_make_bind_گروه_مدخل(binding, ggml_webgpu_tensor_buf(tensor),
                                             ggml_webgpu_tensor_align_جابه_جایی(ctx, tensor),
                                             ggml_webgpu_tensor_binding_اندازه(ctx, tensor));
}

/** End WebGPU object initializations */

/** WebGPU Actions */

قالب <نام_نوع T>
ساكن فراغ ggml_backend_webgpu_check_wait_وضعیت(wgpu::Wait_وضعیت wait_وضعیت,
                                                  T                callback_وضعیت,
                                                  T                success_وضعیت,
                                                  ثابت حرف *     wait_name,
                                                  ثابت حرف *     failure_name,
                                                  ثابت حرف *     callback_پیام) {
    إذا (wait_وضعیت == wgpu::Wait_وضعیت::Timed_بیرون) {
        GGML_ABORT("ggml_webgpu: %s timed out after %u ms\n", wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    إذا (wait_وضعیت == wgpu::Wait_وضعیت::Error) {
        GGML_ABORT("ggml_webgpu: %s failed\n", wait_name);
    }
    إذا (callback_وضعیت != success_وضعیت) {
        GGML_ABORT("ggml_webgpu: %s failed with status %d: %s\n", failure_name, static_cast<صحيح>(callback_وضعیت),
                   callback_پیام);
    }
}

// TODO: these next two functions may want tuning across different platforms and workloads,
ساكن uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    أرجع UINT32_MAX;
}

ساكن uint32_t ggml_backend_webgpu_get_command_submit_batch_اندازه() {
    أرجع WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_اندازه;
}

ساكن فراغ ggml_backend_webgpu_wait_صف(webgpu_global_context & ctx) {
    wgpu::صف_Work_انجامشده_وضعیت callback_وضعیت = wgpu::صف_Work_انجامشده_وضعیت::Error;
    std::string               callback_پیام;

    ثابت wgpu::Wait_وضعیت wait_وضعیت = ctx->instance.WaitAny(
        ctx->queue.On_Submitted_Work_انجامشده(
            wgpu::Callback_نما::AllowSpontaneous,
            [&callback_وضعیت, &callback_پیام](wgpu::صف_Work_انجامشده_وضعیت وضعیت, wgpu::StringView پیام) {
                callback_وضعیت  = وضعیت;
                callback_پیام = std::string(پیام);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_وضعیت(wait_وضعیت, callback_وضعیت, wgpu::صف_Work_انجامشده_وضعیت::Success,
                                          "Queue wait", "Queue work", callback_پیام.c_str());
}

ساكن فراغ ggml_backend_webgpu_map_میانگیر(webgpu_global_context & ctx,
                                           wgpu::میانگیر_2 &          میانگیر,
                                           wgpu::Map_نما           mode,
                                           size_t                  جابه_جایی,
                                           size_t                  size) {
    wgpu::Map_Async_وضعیت callback_وضعیت = wgpu::Map_Async_وضعیت::Error;
    std::string          callback_پیام;

    ثابت wgpu::Wait_وضعیت wait_وضعیت = ctx->instance.WaitAny(
        میانگیر.MapAsync(mode, جابه_جایی, size, wgpu::Callback_نما::AllowSpontaneous,
                        [&callback_وضعیت, &callback_پیام](wgpu::Map_Async_وضعیت وضعیت, wgpu::StringView پیام) {
                            callback_وضعیت  = وضعیت;
                            callback_پیام = std::string(پیام);
                        }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);

    ggml_backend_webgpu_check_wait_وضعیت(wait_وضعیت, callback_وضعیت, wgpu::Map_Async_وضعیت::Success,
                                          "Buffer map wait", "Buffer map", callback_پیام.c_str());
}

ساكن فراغ ggml_backend_webgpu_submit_commands(webgpu_context &          ctx,
                                                ثابت wgpu::Command_میانگیر commands,
                                                uint32_t &                رقم_inflight_batches) {
    إذا (رقم_inflight_batches >= ctx->global_ctx->max_inflight_batches) {
        ggml_backend_webgpu_wait_صف(ctx->global_ctx);
        رقم_inflight_batches = 0;
    }

    ctx->global_ctx->queue.Submit(1, &commands);
    رقم_inflight_batches++;
}

#اگر_تعریف_شده GGML_WEBGPU_DEBUG
// This function adds debugging information to shaders, as WebGPU does not support printing directly.
// To use, add a bind group entry to the setup for the shader you are debugging, add the buffer and
// debug statements in the shader, and then call this function after encoding the commands and submitting them.
ساكن فراغ ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
    encoder.Copy_میانگیر_مقصد_میانگیر(ctx->debug_dev_buf, 0, ctx->debug_host_buf, 0, ctx->debug_host_buf.Get_اندازه());
    wgpu::Command_میانگیر commands = encoder.پایان();
    ctx->queue.Submit(1, &commands);
    ggml_backend_webgpu_map_میانگیر(ctx, ctx->debug_host_buf, wgpu::Map_نما::Read, 0, ctx->debug_host_buf.Get_اندازه());
    ثابت عائم * debug_داده = (ثابت عائم *) ctx->debug_host_buf.GetConstMappedRange();
    std::خروجی << "debug[0]: " << debug_داده[0] << "\n";
    ctx->debug_host_buf.Unmap();
}
#پایان_اگر

ساكن webgpu_encoded_op ggml_backend_webgpu_build_multi(webgpu_context &                          ctx,
                                                         ثابت std::vector<webgpu_dispatch_desc> & dispatches) {
    webgpu_encoded_op            نتیجه = {};
    std::vector<wgpu::Bind_گروه> bind_groups;
    std::vector<size_t>          param_offsets;
    نتیجه.رقم_kernels = dispatches.size();

    لكل (size_t i = 0; i < dispatches.size(); i++) {
        ثابت webgpu_dispatch_desc & dispatch     = dispatches[i];
        ثابت size_t                 param_اندازه   = dispatch.params.size() * حجم(uint32_t);
        ثابت size_t                 param_جابه_جایی = ctx->param_arena.alloc_جایگاه(param_اندازه);

        std::vector<wgpu::Bind_گروه_مدخل> entries            = dispatch.bind_گروه_entries;
        uint32_t                          params_binding_رقم = entries.size();
        entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(params_binding_رقم, ctx->param_arena.میانگیر, param_جابه_جایی,
                                                            ctx->param_arena.جایگاه_اندازه));

        wgpu::Bind_گروه_Descriptor bind_گروه_desc;
        bind_گروه_desc.layout     = dispatch.pipeline.pipeline.Get_Bind_گروه_Layout(0);
        bind_گروه_desc.مدخل_شمار = entries.size();
        bind_گروه_desc.entries    = entries.داده();
        bind_گروه_desc.label      = dispatch.pipeline.name.c_str();
        bind_groups.push_back(ctx->global_ctx->device.Create_Bind_گروه(&bind_گروه_desc));
        param_offsets.push_back(param_جابه_جایی);
    }

    لكل (size_t i = 0; i < param_offsets.size(); i++) {
        ctx->global_ctx->queue.Write_میانگیر(ctx->param_arena.میانگیر, param_offsets[i], dispatches[i].params.داده(),
                                           dispatches[i].params.size() * حجم(uint32_t));
    }

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    لكل (size_t i = 0; i < dispatches.size(); i++) {
        GGML_ASSERT(ctx->profile_timestamp_query_شمار + 2 <= WEBGPU_MAX_PROFILE_QUERY_شمار);
        ثابت uint32_t query_begin = ctx->profile_timestamp_query_شمار++;
        ثابت uint32_t query_انتها   = ctx->profile_timestamp_query_شمار++;

        wgpu::PassTimestampWrites ts_writes   = {};
        ts_writes.query_مجموعه                    = ctx->profile_timestamp_query_مجموعه;
        ts_writes.beginning_از_Pass_Write_شاخص   = query_begin;
        ts_writes.انتها_از_Pass_Write_شاخص         = query_انتها;
        wgpu::ComputePassDescriptor pass_desc = {};
        pass_desc.timestampWrites             = &ts_writes;

        wgpu::ComputePassEncoder pass = ctx->فعال_command_encoder.BeginComputePass(&pass_desc);

        pass.مجموعه_Pipeline(dispatches[i].pipeline.pipeline);
        pass.مجموعه_Bind_گروه(0, bind_groups[i]);
        pass.DispatchWorkgroups(dispatches[i].workgroups.اولی, dispatches[i].workgroups.second, 1);
        pass.انتها_2();
        نتیجه.pipeline_نامها.push_back(dispatches[i].pipeline.name);
    }
#وإلا
    لكل (size_t i = 0; i < dispatches.size(); i++) {
        إذا (ctx->batch_compute_passes) {
            ctx->فعال_compute_pass.مجموعه_Pipeline(dispatches[i].pipeline.pipeline);
            ctx->فعال_compute_pass.مجموعه_Bind_گروه(0, bind_groups[i]);
            ctx->فعال_compute_pass.DispatchWorkgroups(dispatches[i].workgroups.اولی, dispatches[i].workgroups.second,
                                                        1);
        } وإلا {
            wgpu::ComputePassEncoder pass = ctx->فعال_command_encoder.BeginComputePass();
            pass.مجموعه_Pipeline(dispatches[i].pipeline.pipeline);
            pass.مجموعه_Bind_گروه(0, bind_groups[i]);
            pass.DispatchWorkgroups(dispatches[i].workgroups.اولی, dispatches[i].workgroups.second, 1);
            pass.انتها_2();
        }
    }
#پایان_اگر

    أرجع نتیجه;
}

ساكن webgpu_encoded_op ggml_backend_webgpu_build(webgpu_context &                  ctx,
                                                   webgpu_pipeline &                 pipeline,
                                                   std::vector<uint32_t>             params,
                                                   std::vector<wgpu::Bind_گروه_مدخل> bind_گروه_entries,
                                                   uint32_t                          wg_x,
                                                   uint32_t                          wg_y = 1) {
    أرجع ggml_backend_webgpu_build_multi(
        ctx, {
                 { pipeline, std::move(params), std::move(bind_گروه_entries), { wg_x, wg_y } },
    });
}

ساكن فراغ ggml_backend_webgpu_میانگیر_memset(webgpu_global_context & ctx,
                                              wgpu::میانگیر_2 &          buf,
                                              uint32_t                value,
                                              size_t                  جابه_جایی,
                                              size_t                  size) {
    std::vector<uint32_t>             params  = { (uint32_t) جابه_جایی, (uint32_t) size, value };
    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_bind_گروه_مدخل(0, buf, 0, buf.Get_اندازه()) };
    size_t                            بایتها_در_هر_wg =
        ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup * ctx->capabilities.memset_بایتها_در_هر_thread;
    uint32_t wg_x = CEIL_DIV(size + 3, بایتها_در_هر_wg);

    ctx->queue.Write_میانگیر(ctx->memset_params_buf, 0, params.داده(), params.size() * حجم(uint32_t));

    wgpu::Bind_گروه_مدخل params_مدخل = {};
    params_مدخل.binding              = 1;
    params_مدخل.میانگیر               = ctx->memset_params_buf;
    params_مدخل.جابه_جایی               = 0;
    params_مدخل.size                 = WEBGPU_PARAMS_BUF_اندازه_بایتها;
    entries.push_back(params_مدخل);

    wgpu::Bind_گروه_Descriptor bind_گروه_desc;
    bind_گروه_desc.layout     = ctx->memset_pipeline.pipeline.Get_Bind_گروه_Layout(0);
    bind_گروه_desc.مدخل_شمار = entries.size();
    bind_گروه_desc.entries    = entries.داده();
    bind_گروه_desc.label      = ctx->memset_pipeline.name.c_str();
    wgpu::Bind_گروه bind_گروه = ctx->device.Create_Bind_گروه(&bind_گروه_desc);

    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
    pass.مجموعه_Pipeline(ctx->memset_pipeline.pipeline);
    pass.مجموعه_Bind_گروه(0, bind_گروه);
    pass.DispatchWorkgroups(wg_x, 1, 1);
    pass.انتها_2();

    wgpu::Command_میانگیر              command  = encoder.پایان();
    std::vector<wgpu::Command_میانگیر> commands = { command };
    ctx->queue.Submit(commands.size(), commands.داده());
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

#اگر_تعریف_شده GGML_WEBGPU_CPU_PROFILE
    std::خروجی << "\n[ggml_webgpu cpu profiling summary]\n";
    مزدوج مجموع_cpu = 0.0;
    لكل (ثابت خودکار & kv : ctx->webgpu_ctx->global_ctx->cpu_زمان_ms) {
        مجموع_cpu += kv.second;
    }
    std::خروجی << "ggml_webgpu: total cpu time: " << مجموع_cpu << " ms\n";
    std::خروجی << "ggml_webgpu: cpu breakdown:\n";
    لكل (ثابت خودکار & kv : ctx->webgpu_ctx->global_ctx->cpu_زمان_ms) {
        مزدوج pct = (مجموع_cpu > 0.0) ? (kv.second / مجموع_cpu * 100.0) : 0.0;
        std::خروجی << "ggml_webgpu:  " << kv.اولی << ": " << kv.second << " ms (" << pct << "%)\n";
    }
    إذا (ctx->webgpu_ctx->global_ctx->cpu_detail_ms.size() > 0) {
        std::خروجی << "ggml_webgpu: cpu detailed breakdown:\n";
    }
    لكل (ثابت خودکار & kv : ctx->webgpu_ctx->global_ctx->cpu_detail_ms) {
        مزدوج pct = (مجموع_cpu > 0.0) ? (kv.second / مجموع_cpu * 100.0) : 0.0;
        std::خروجی << "ggml_webgpu:  " << kv.اولی << ": " << kv.second << " ms (" << pct << "%)\n";
    }
#پایان_اگر

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    std::خروجی << "\n[ggml_webgpu gpu profiling summary]\n";
    مزدوج مجموع_gpu = 0.0;
    لكل (ثابت خودکار & kv : ctx->webgpu_ctx->shader_gpu_زمان_ms) {
        مجموع_gpu += kv.second;
    }
    std::خروجی << "ggml_webgpu: total gpu time (all shaders): " << مجموع_gpu << " ms\n";
    std::خروجی << "\nggml_webgpu: gpu breakdown:\n";
    لكل (ثابت خودکار & kv : ctx->webgpu_ctx->shader_gpu_زمان_ms) {
        مزدوج pct = (مجموع_gpu > 0.0) ? (kv.second / مجموع_gpu * 100.0) : 0.0;
        std::خروجی << "ggml_webgpu:  " << kv.اولی << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                  << pct << "%)\n";
    }
#پایان_اگر

#إذا defined(GGML_WEBGPU_CPU_PROFILE) && defined(GGML_WEBGPU_GPU_PROFILE)
    std::خروجی << "ggml_webgpu: gpu/cpu ratio: " << (مجموع_cpu > 0.0 ? مجموع_gpu / مجموع_cpu : 0.0) << "\n";
#پایان_اگر

    حذف ctx;
    حذف backend;
}

ساكن webgpu_encoded_op ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cpy_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_مجموعه(webgpu_context & ctx,
                                         ggml_tensor *    src0,
                                         ggml_tensor *    src1,
                                         ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_مجموعه_pipeline(shader_lib_ctx);

    خودکار *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
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

    std::vector<wgpu::Bind_گروه_مدخل> entries;
    uint32_t                          binding_شاخص = 0;
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0));
        binding_شاخص++;
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_شاخص, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_شاخص + 1, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_pad_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst),
    };

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
    shader_lib_ctx.wg_mem_حد_بایتها = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_اندازه;

    webgpu_pipeline pipeline = ctx->shader_lib->get_solve_tri_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst),
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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);

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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    // Input layout: contiguous -> WHCN, contiguous-channels -> CWHN
    ثابت منطقي      whcn      = ggml_is_contiguous(src1);
    webgpu_pipeline pipeline  = ctx->shader_lib->get_conv2d_dw_pipeline(shader_lib_ctx, whcn);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);

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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src1),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);

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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_conv_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_ssm_conv_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t token_tiles = CEIL_DIV((uint32_t) dst->ne[1], decisions->نشانهها_در_هر_wg);

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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst),
    };

    ثابت uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->بلوک_اندازه_3);
    ثابت uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_ssm_پیمایش(webgpu_context & ctx,
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
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
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

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ssm_پیمایش_pipeline(shader_lib_ctx, xbc_overlap);
    خودکار *          decisions = static_cast<ggml_webgpu_ssm_پیمایش_shader_decisions *>(pipeline.context.get());
    xbc_overlap               = decisions->xbc_overlap;

    uint32_t جابه_جایی_x   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    uint32_t جابه_جایی_dt  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type));
    uint32_t جابه_جایی_A   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_اندازه(src3->type));
    uint32_t جابه_جایی_B   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_اندازه(src4->type));
    uint32_t جابه_جایی_C   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_اندازه(src5->type));
    uint32_t جابه_جایی_ids = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_اندازه(src6->type));

    ggml_webgpu_merged_binding_range xbc_merged_range = {};

    إذا (xbc_overlap == 0b110) {  // x/B
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4 });
        جابه_جایی_x         = ggml_webgpu_tensor_merged_element_جابه_جایی(src1, xbc_merged_range);
        جابه_جایی_B         = ggml_webgpu_tensor_merged_element_جابه_جایی(src4, xbc_merged_range);
    } وإلا إذا (xbc_overlap == 0b011) {  // B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src4, src5 });
        جابه_جایی_B         = ggml_webgpu_tensor_merged_element_جابه_جایی(src4, xbc_merged_range);
        جابه_جایی_C         = ggml_webgpu_tensor_merged_element_جابه_جایی(src5, xbc_merged_range);
    } وإلا إذا (xbc_overlap == 0b111) {  // x/B/C
        xbc_merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
        جابه_جایی_x         = ggml_webgpu_tensor_merged_element_جابه_جایی(src1, xbc_merged_range);
        جابه_جایی_B         = ggml_webgpu_tensor_merged_element_جابه_جایی(src4, xbc_merged_range);
        جابه_جایی_C         = ggml_webgpu_tensor_merged_element_جابه_جایی(src5, xbc_merged_range);
    }

    GGML_ASSERT(xbc_overlap == 0 || xbc_overlap == 0b110 || xbc_overlap == 0b011 || xbc_overlap == 0b111);

    size_t xbc_bind_جابه_جایی = 0;
    size_t xbc_bind_اندازه   = 0;
    إذا (xbc_overlap > 0) {
        xbc_bind_جابه_جایی = xbc_merged_range.جابه_جایی;
        xbc_bind_اندازه   = xbc_merged_range.size;
    }

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        جابه_جایی_x,
        جابه_جایی_dt,
        جابه_جایی_A,
        جابه_جایی_B,
        جابه_جایی_C,
        جابه_جایی_ids,
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

    uint32_t binding_رقم = 0;

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, src0),
    };
    // xbc_merged binding
    إذا (xbc_overlap > 0) {
        entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(binding_رقم++, ggml_webgpu_tensor_buf(src1),
                                                            xbc_bind_جابه_جایی, xbc_bind_اندازه));
    }
    // x
    إذا (!(xbc_overlap & 0b100)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, src1));
    }
    // B
    إذا (!(xbc_overlap & 0b010)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, src4));
    }
    // C
    إذا (!(xbc_overlap & 0b001)) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, src5));
    }

    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, src2));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, src3));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, src6));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم++, dst));

    ثابت uint32_t مجموع_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
    ثابت uint32_t max_wg_در_هر_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension;
    uint32_t       wg_x;
    uint32_t       wg_y;
    compute_2d_workgroups(مجموع_wg, max_wg_در_هر_dim, wg_x, wg_y);

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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_gated_delta_net_pipeline(shader_lib_ctx);

    ثابت uint32_t s_v      = (uint32_t) src2->ne[0];
    ثابت uint32_t h        = (uint32_t) src2->ne[1];
    ثابت uint32_t n_نشانهها = (uint32_t) src2->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t) src2->ne[3];
    ثابت uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);
    ثابت عائم    scale    = 1.0f / sqrtf((عائم) s_v);
    uint32_t       scale_u32;
    کپی_حافظه(&scale_u32, &scale, حجم(scale_u32));

    std::vector<uint32_t> params = {
        h,
        n_نشانهها,
        n_seqs,
        s_v * h * n_نشانهها * n_seqs,

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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0), ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, src2), ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 3, src3),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 4, src4), ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 5, src5),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 6, dst),
    };

    إذا (dst_fuse) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 7, dst_fuse));
    }

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, h, n_seqs);
}

ساكن std::optional<webgpu_encoded_op> ggml_webgpu_مجموعه_ردیفها(webgpu_context & ctx,
                                                             ggml_tensor *    src,
                                                             ggml_tensor *    idx,
                                                             ggml_tensor *    dst) {
    // For set rows specifically, we need to check if src and idx are empty
    // tensors.
    إذا (ggml_is_خالی(src) || ggml_is_خالی(idx)) {
        أرجع std::nullopt;
    }

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = idx;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_مجموعه_ردیفها_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_مجموعه_ردیفها_shader_decisions *>(pipeline.context.get());

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

    std::vector<wgpu::Bind_گروه_مدخل> entries;
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, idx));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst));

    إذا (decisions->i64_idx) {
        entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(3, ctx->مجموعه_ردیفها_dev_error_buf, 0,
                                                            ctx->مجموعه_ردیفها_dev_error_buf.Get_اندازه()));
    }

    uint32_t نخها;
    إذا (ggml_is_quantized(dst->type)) {
        ثابت uint32_t بلوکها_در_هر_ردیف = src->ne[0] / ggml_blck_اندازه(dst->type);
        نخها =
            (src->ne[1] * src->ne[2] * src->ne[3]) * (decisions->pair_بلوکها ? (بلوکها_در_هر_ردیف / 2) : بلوکها_در_هر_ردیف);
    } وإلا إذا (decisions->vec4) {
        نخها = (src->ne[1] * src->ne[2] * src->ne[3]) * (src->ne[0] / 4);
    } وإلا {
        نخها = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    }
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(نخها, decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Workgroup size is a common constant
ساكن std::vector<wgpu::Constant_مدخل> ggml_webgpu_wg_اندازه_مدخل(uint32_t wg_اندازه) {
    std::vector<wgpu::Constant_مدخل> constants(1);
    constants[0].کلید   = "wg_size";
    constants[0].value = wg_اندازه;
    أرجع constants;
}

ساكن webgpu_encoded_op ggml_webgpu_get_ردیفها(webgpu_context & ctx,
                                              ggml_tensor *    src,
                                              ggml_tensor *    idx,
                                              ggml_tensor *    dst) {
    ثابت منطقي float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = اشاره_تهی;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    ثابت uint32_t جابه_جایی_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type));
    ثابت uint32_t جابه_جایی_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type));
    ثابت uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type));

    ثابت منطقي vec4_aligned = جابه_جایی_src % 4 == 0 && جابه_جایی_dst % 4 == 0 && stride_src1 % 4 == 0 &&
                              stride_src2 % 4 == 0 && stride_src3 % 4 == 0;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_get_ردیفها_pipeline(shader_lib_ctx, vec4_aligned);
    خودکار *          decisions = static_cast<ggml_webgpu_get_ردیفها_shader_decisions *>(pipeline.context.get());

    std::vector<uint32_t> params = { جابه_جایی_src,
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, idx) / ggml_type_اندازه(idx->type)),
                                     جابه_جایی_dst,
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

    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, idx),
                                                  ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst) };

    uint32_t بلوکها_در_هر_ردیف = (uint32_t) (dst->ne[0] / (decisions->vectorized ? 4 : 1));
    uint32_t مجموع_ردیفها     = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    uint32_t مجموع_نخها  = float_parallel ? بلوکها_در_هر_ردیف * مجموع_ردیفها : مجموع_ردیفها;
    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(مجموع_نخها, decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;

    webgpu_pipeline qq8_pipeline = ctx->shader_lib->get_quantize_q8_pipeline(shader_lib_ctx);

    // quantize_q8 pipeline
    ثابت size_t dst_جابه_جایی           = ggml_webgpu_tensor_جابه_جایی(dst);
    ثابت size_t q8_src1_align_جابه_جایی = ROUNDUP_POW2(
        dst_جابه_جایی + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
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

    std::vector<wgpu::Bind_گروه_مدخل> q8_entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src1),
        ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(dst), q8_src1_align_جابه_جایی, q8_src1_binding_اندازه)
    };

    خودکار q8_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(qq8_pipeline.context.get());

    uint32_t       q8_wg_اندازه     = q8_decisions->wg_اندازه;
    uint32_t       q8_wg_x        = 1;
    uint32_t       q8_wg_y        = 1;
    ثابت uint32_t wg_در_هر_vec     = (src0->ne[0] / 4 + (q8_wg_اندازه - 1)) / q8_wg_اندازه;
    ثابت uint32_t q8_مجموع_wg    = src1->ne[1] * src1->ne[2] * src1->ne[3] * wg_در_هر_vec;
    ثابت uint32_t max_wg_در_هر_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension;
    compute_2d_workgroups(q8_مجموع_wg, max_wg_در_هر_dim, q8_wg_x, q8_wg_y);

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
    منطقي use_mmvq = ggml_webgpu_can_use_mmvq(src0, src1, ctx->global_ctx->capabilities.supports_dot_حاصلضرب,
                                             ctx->global_ctx->vendor);

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};

    shader_lib_ctx.src0                     = src0;
    shader_lib_ctx.src1                     = src1;
    shader_lib_ctx.dst                      = dst;
    shader_lib_ctx.max_wg_اندازه              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
    shader_lib_ctx.supports_subgroups       = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.supports_subgroup_ماتریس = ctx->global_ctx->capabilities.supports_subgroup_ماتریس;
    shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    shader_lib_ctx.min_subgroup_اندازه        = ctx->global_ctx->capabilities.min_subgroup_اندازه;
    shader_lib_ctx.max_subgroup_اندازه        = ctx->global_ctx->capabilities.max_subgroup_اندازه;
    shader_lib_ctx.supports_dot_حاصلضرب     = ctx->global_ctx->capabilities.supports_dot_حاصلضرب;
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

    uint32_t جابه_جایی_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t جابه_جایی_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    size_t   merged_جابه_جایی = 0;
    size_t   merged_اندازه   = 0;
    إذا (src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_جابه_جایی = merged_range.جابه_جایی;
        merged_اندازه   = merged_range.size;
        جابه_جایی_src0   = ggml_webgpu_tensor_merged_element_جابه_جایی(src0, merged_range);
        جابه_جایی_src1   = ggml_webgpu_tensor_merged_element_جابه_جایی(src1, merged_range);
    }

    // Build params
    std::vector<uint32_t> params = { جابه_جایی_src0,
                                     جابه_جایی_src1,
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
    std::vector<wgpu::Bind_گروه_مدخل> entries = {};
    إذا (use_mmvq) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0));
        خودکار & mmvq_qq8_مدخل = dispatches[0].bind_گروه_entries[1];
        entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_مدخل.جابه_جایی,
                                                            mmvq_qq8_مدخل.size));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst));
    } وإلا إذا (src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src0), merged_جابه_جایی, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst));
    }

    // Calculate workgroup dimensions
    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    ثابت uint32_t max_wg_در_هر_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension;

    إذا (use_mat_vec) {
        خودکار * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

        uint32_t batches       = dst->ne[2] * dst->ne[3];
        uint32_t output_groups = CEIL_DIV(dst->ne[0], decisions->outputs_در_هر_wg);
        uint32_t مجموع_wg      = output_groups * batches;
        compute_2d_workgroups(مجموع_wg, max_wg_در_هر_dim, wg_x, wg_y);
    } وإلا {
        خودکار * decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(pipeline.context.get());

        // Fast-path tiled/subgroup calculations
        uint32_t wg_m;
        uint32_t wg_n;
        إذا (decisions->use_subgroup_ماتریس) {
            uint32_t wg_m_sg_tile =
                decisions->subgroup_m * decisions->subgroup_ماتریس_m * ctx->global_ctx->capabilities.sg_mat_m;
            wg_m = CEIL_DIV(dst->ne[0], wg_m_sg_tile);
            uint32_t wg_n_sg_tile =
                decisions->subgroup_n * decisions->subgroup_ماتریس_n * ctx->global_ctx->capabilities.sg_mat_n;
            wg_n = CEIL_DIV(dst->ne[1], wg_n_sg_tile);
        } وإلا {
            uint32_t tile_m_s = decisions->tile_m * decisions->wg_اندازه_m;
            uint32_t tile_n_s = decisions->tile_n * decisions->wg_اندازه_n;
            wg_m              = CEIL_DIV(dst->ne[0], tile_m_s);
            wg_n              = CEIL_DIV(dst->ne[1], tile_n_s);
        }
        uint32_t مجموع_wg = wg_m * wg_n * dst->ne[2] * dst->ne[3];
        compute_2d_workgroups(مجموع_wg, max_wg_در_هر_dim, wg_x, wg_y);
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
    ثابت uint32_t param_n_expert_استفاده_شده = (uint32_t) dst->ne[1];

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_mul_mat_id_vec_pipeline(shader_lib_ctx);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) src0->ne[0],
        (uint32_t) src0->ne[1],
        param_n_expert,
        param_n_expert_استفاده_شده,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
    };

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_جابه_جایی(ctx, src0),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src0)),
        ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_جابه_جایی(ctx, src1),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src1)),
        ggml_webgpu_make_bind_گروه_مدخل(2, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_جابه_جایی(ctx, src2),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src2)),
        ggml_webgpu_make_bind_گروه_مدخل(3, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_جابه_جایی(ctx, dst),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, dst)),
    };

    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    خودکار * decisions = static_cast<ggml_webgpu_mul_mat_vec_shader_decisions *>(pipeline.context.get());

    ثابت uint32_t max_wg_در_هر_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension;
    uint32_t       output_groups  = CEIL_DIV(dst->ne[0], decisions->outputs_در_هر_wg);
    uint32_t       مجموع_wg       = output_groups * param_n_expert_استفاده_شده;
    compute_2d_workgroups(مجموع_wg, max_wg_در_هر_dim, wg_x, wg_y);

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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    // Get or create pipeline
    webgpu_pipeline gather_pipeline;
    webgpu_pipeline main_pipeline;

    std::vector<webgpu_dispatch_desc> dispatches;

    gather_pipeline = ctx->shader_lib->get_mul_mat_id_gather_pipeline(shader_lib_ctx);
    main_pipeline   = ctx->shader_lib->get_mul_mat_id_pipeline(shader_lib_ctx);

    ثابت uint32_t param_n_expert      = (uint32_t) src0->ne[2];
    ثابت uint32_t param_n_expert_استفاده_شده = (uint32_t) dst->ne[1];
    ثابت uint32_t param_n_نشانهها      = (uint32_t) dst->ne[2];

    // params for mul_mat_id_gather.wgsl
    std::vector<uint32_t> gather_params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)),
        param_n_expert,
        param_n_expert_استفاده_شده,
        param_n_نشانهها,
        (uint32_t) (src2->nb[1] / ggml_type_اندازه(src2->type)),
    };

    ثابت size_t dst_جابه_جایی       = ggml_webgpu_tensor_جابه_جایی(dst);
    ثابت size_t ids_cache_nbytes = src2->ne[0] * src1->ne[2] * حجم(uint32_t);
    ثابت size_t ids_tiles_nbytes = ggml_webgpu_mul_mat_id_tiles_nbytes(dst);

    ثابت size_t ids_cache_align_جابه_جایی = ROUNDUP_POW2(
        dst_جابه_جایی + ggml_nbytes(dst), ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
    ثابت size_t ids_tiles_align_جابه_جایی =
        ROUNDUP_POW2(ids_cache_align_جابه_جایی + ids_cache_nbytes,
                     ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment);

    ثابت size_t ids_cache_bind_اندازه = ROUNDUP_POW2(ids_cache_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ثابت size_t ids_tiles_bind_اندازه = ROUNDUP_POW2(ids_tiles_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);

    // bind group entries for mul_mat_id_gather.wgsl
    std::vector<wgpu::Bind_گروه_مدخل> gather_entries = {
        ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src2), ggml_webgpu_tensor_align_جابه_جایی(ctx, src2),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src2)),
        ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(dst), ids_cache_align_جابه_جایی, ids_cache_bind_اندازه),
        ggml_webgpu_make_bind_گروه_مدخل(2, ggml_webgpu_tensor_buf(dst), ids_tiles_align_جابه_جایی, ids_tiles_bind_اندازه),
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
        param_n_expert_استفاده_شده,
        param_n_نشانهها,
        (uint32_t) src1->ne[1],
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)),
    };

    // bind group entries for mul_mat_id.wgsl
    std::vector<wgpu::Bind_گروه_مدخل> main_entries = {
        ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_جابه_جایی(ctx, src0),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src0)),
        ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(src1), ggml_webgpu_tensor_align_جابه_جایی(ctx, src1),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, src1)),
        ggml_webgpu_make_bind_گروه_مدخل(2, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_جابه_جایی(ctx, dst),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, dst)),
        ggml_webgpu_make_bind_گروه_مدخل(3, ggml_webgpu_tensor_buf(dst), ids_cache_align_جابه_جایی, ids_cache_bind_اندازه),
        ggml_webgpu_make_bind_گروه_مدخل(4, ggml_webgpu_tensor_buf(dst), ids_tiles_align_جابه_جایی, ids_tiles_bind_اندازه),
    };

    // Calculate workgroup dimensions
    uint32_t wg_x = 1;
    uint32_t wg_y = 1;

    خودکار * main_decisions = static_cast<ggml_webgpu_mul_mat_shader_decisions *>(main_pipeline.context.get());

    // must match `TILE_N_SLOTS` of mul_mat_id_gather.wgsl
    GGML_ASSERT(main_decisions->tile_n * main_decisions->wg_اندازه_n == WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_اندازه_N);

    ثابت uint32_t tile_m_s = main_decisions->tile_m * main_decisions->wg_اندازه_m;
    ثابت uint32_t wg_m     = CEIL_DIV(dst->ne[0], tile_m_s);
    ثابت uint32_t مجموع_wg = wg_m * ggml_webgpu_mul_mat_id_max_tiles(dst);

    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);

    dispatches.push_back({
        main_pipeline, std::move(main_params), std::move(main_entries), { wg_x, wg_y }
    });

    أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
}

بنية ggml_webgpu_flash_attn_op {
    ggml_webgpu_shader_lib_context    shader_lib_ctx = {};
    std::vector<uint32_t>             params;
    std::vector<wgpu::Bind_گروه_مدخل> entries;
    size_t                            kv_bind_جابه_جایی = 0;
    size_t                            kv_bind_اندازه   = 0;
    منطقي                              has_ماسک       = خطأ;
    منطقي                              has_sinks      = خطأ;
    منطقي                              kv_overlap     = خطأ;
};

ساكن منطقي ggml_webgpu_flash_attn_use_vec_مسیر(ثابت webgpu_global_context & global_ctx,
                                                ثابت ggml_tensor *           Q,
                                                ثابت ggml_tensor *           K,
                                                ثابت ggml_tensor *           V) {
    ثابت size_t storage_جابه_جایی_alignment = global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment;

    ثابت منطقي k_float_vec4_aligned = (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(K, storage_جابه_جایی_alignment);
    ثابت منطقي v_float_vec4_aligned = (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) ||
                                      ggml_webgpu_flash_attn_float_vec4_aligned(V, storage_جابه_جایی_alignment);

    ثابت uint32_t k_vec_سر_align =
        ggml_is_quantized(K->type) ? ggml_blck_اندازه(K->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض;
    ثابت uint32_t v_vec_سر_align =
        ggml_is_quantized(V->type) ? ggml_blck_اندازه(V->type) : GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض;
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
    op.shader_lib_ctx.supports_subgroup_ماتریس = ctx->global_ctx->capabilities.supports_subgroup_ماتریس;
    op.shader_lib_ctx.max_wg_اندازه              = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
    op.shader_lib_ctx.wg_mem_حد_بایتها       = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_اندازه;
    op.shader_lib_ctx.sg_mat_m                 = ctx->global_ctx->capabilities.sg_mat_m;
    op.shader_lib_ctx.sg_mat_n                 = ctx->global_ctx->capabilities.sg_mat_n;
    op.shader_lib_ctx.sg_mat_k                 = ctx->global_ctx->capabilities.sg_mat_k;
    op.shader_lib_ctx.min_subgroup_اندازه        = ctx->global_ctx->capabilities.min_subgroup_اندازه;
    op.shader_lib_ctx.max_subgroup_اندازه        = ctx->global_ctx->capabilities.max_subgroup_اندازه;

    op.has_ماسک   = ماسک != اشاره_تهی;
    op.has_sinks  = sinks != اشاره_تهی;
    op.kv_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, K, V);

    uint32_t جابه_جایی_k = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_اندازه(K->type));
    uint32_t جابه_جایی_v = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_اندازه(V->type));
    إذا (op.kv_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
        op.kv_bind_جابه_جایی                                   = merged_range.جابه_جایی;
        op.kv_bind_اندازه                                     = merged_range.size;
        جابه_جایی_k                                            = ggml_webgpu_tensor_merged_element_جابه_جایی(K, merged_range);
        جابه_جایی_v                                            = ggml_webgpu_tensor_merged_element_جابه_جایی(V, merged_range);
    }

    op.params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_اندازه(Q->type)),
        جابه_جایی_k,
        جابه_جایی_v,
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
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, Q),
    };
    إذا (op.kv_overlap) {
        op.entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(K), op.kv_bind_جابه_جایی, op.kv_bind_اندازه));
    } وإلا {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, K));
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, V));
    }
    uint32_t binding_شاخص = op.kv_overlap ? 2u : 3u;
    إذا (op.has_ماسک) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_شاخص++, ماسک));
    }
    إذا (op.has_sinks) {
        op.entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_شاخص++, sinks));
    }
    op.entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_شاخص++, dst));

    أرجع op;
}

ساكن uint32_t ggml_webgpu_flash_attn_vec_nwg(uint32_t vec_nwg_cap, uint32_t kv_tile, uint32_t seq_len_kv) {
    uint32_t       nwg     = 1u;
    ثابت uint64_t kv_بازه = (uint64_t) kv_tile;
    طالما ((2u * nwg * kv_بازه) < (uint64_t) seq_len_kv && nwg < vec_nwg_cap) {
        nwg <<= 1;
    }
    أرجع std::min(nwg, vec_nwg_cap);
}

ساكن webgpu_encoded_op ggml_webgpu_flash_attn_مستقیم(webgpu_context & ctx, ثابت ggml_webgpu_flash_attn_op & op) {
    webgpu_pipeline pipeline    = ctx->shader_lib->get_flash_attn_pipeline(op.shader_lib_ctx, op.kv_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
    uint32_t        wg_در_هر_سر = CEIL_DIV(op.shader_lib_ctx.src0->ne[1], decisions->q_tile);
    uint32_t        wg_x        = wg_در_هر_سر * op.shader_lib_ctx.src0->ne[2] * op.shader_lib_ctx.src0->ne[3];
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
    خودکار *          decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());

    wgpu::میانگیر_2 blk_buf         = {};
    uint64_t     blk_اندازه_بایتها  = 0;
    uint32_t     blk_nblk0       = 0;
    uint32_t     blk_nblk1       = 0;
    uint32_t     blk_batch_شمار = 0;

    ثابت uint32_t vec_nwg_cap = ctx->global_ctx->capabilities.min_subgroup_اندازه;
    uint32_t       nwg         = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, decisions->kv_tile, (uint32_t) K->ne[1]);
    ثابت uint64_t nrows       = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
    ثابت منطقي     use_vec_reduce = nwg > 1u;
    GGML_ASSERT(nrows <= UINT32_MAX);

    uint64_t     tmp_stats_پایه  = 0;
    uint64_t     tmp_اندازه_بایتها  = 0;
    wgpu::میانگیر_2 tmp_buf         = {};
    uint64_t     tmp_bind_جابه_جایی = 0;
    uint64_t     tmp_bind_اندازه   = 0;
    ثابت size_t align_بایتها     = ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment;
    ثابت size_t dst_جابه_جایی      = ggml_webgpu_tensor_جابه_جایی(dst);
    size_t       scratch_جابه_جایی  = ROUNDUP_POW2(dst_جابه_جایی + ggml_nbytes(dst), align_بایتها);

    إذا (use_vec_reduce) {
        ثابت uint64_t tmp_داده_elems  = nrows * (uint64_t) V->ne[0] * nwg;
        ثابت uint64_t tmp_stats_elems = nrows * 2u * nwg;
        tmp_stats_پایه                 = tmp_داده_elems;
        tmp_اندازه_بایتها =
            ROUNDUP_POW2((tmp_داده_elems + tmp_stats_elems) * حجم(عائم), WEBGPU_STORAGE_BUF_BINDING_MULT);
        GGML_ASSERT(tmp_stats_پایه <= UINT32_MAX);
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_جابه_جایی = scratch_جابه_جایی;
        tmp_bind_اندازه   = tmp_اندازه_بایتها;
        scratch_جابه_جایی  = ROUNDUP_POW2(scratch_جابه_جایی + tmp_اندازه_بایتها, align_بایتها);
    } وإلا {
        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
        tmp_اندازه_بایتها  = WEBGPU_STORAGE_BUF_BINDING_MULT;
        tmp_buf         = ggml_webgpu_tensor_buf(dst);
        tmp_bind_جابه_جایی = scratch_جابه_جایی;
        tmp_bind_اندازه   = tmp_اندازه_بایتها;
        scratch_جابه_جایی  = ROUNDUP_POW2(scratch_جابه_جایی + tmp_اندازه_بایتها, align_بایتها);
    }

    webgpu_pipeline                   blk_pipeline;
    std::vector<uint32_t>             blk_params;
    std::vector<wgpu::Bind_گروه_مدخل> blk_entries;
    إذا (op.has_ماسک) {
        blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
        blk_nblk1                   = (uint32_t) Q->ne[1];
        blk_buf                     = ggml_webgpu_tensor_buf(dst);
        ثابت uint32_t stride_mask3 = (uint32_t) (ماسک->nb[3] / ggml_type_اندازه(ماسک->type));
        blk_batch_شمار             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
        ثابت uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_شمار;
        blk_اندازه_بایتها              = ROUNDUP_POW2(blk_elems * حجم(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
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
            ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(ماسک),
                                              ggml_webgpu_tensor_align_جابه_جایی(ctx, ماسک),
                                              ggml_webgpu_tensor_binding_اندازه(ctx, ماسک)),
            ggml_webgpu_make_bind_گروه_مدخل(1, blk_buf, scratch_جابه_جایی, blk_اندازه_بایتها),
        };
        scratch_جابه_جایی = ROUNDUP_POW2(scratch_جابه_جایی + blk_اندازه_بایتها, align_بایتها);
    }

    std::vector<uint32_t> تقسیم_params = op.params;
    إذا (op.has_ماسک) {
        تقسیم_params.push_back(0u);                     // blk_base
        تقسیم_params.push_back(blk_nblk0);              // blk_nblk0
        تقسیم_params.push_back(blk_nblk1);              // blk_nblk1
    }
    تقسیم_params.push_back(0u);                         // tmp_data_base
    تقسیم_params.push_back((uint32_t) tmp_stats_پایه);  // tmp_stats_base
    تقسیم_params.push_back(nwg);                        // nwg

    std::vector<wgpu::Bind_گروه_مدخل> تقسیم_entries = {
        ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_جابه_جایی(ctx, Q),
                                          ggml_webgpu_tensor_binding_اندازه(ctx, Q)),
    };
    إذا (op.kv_overlap) {
        تقسیم_entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(K), op.kv_bind_جابه_جایی, op.kv_bind_اندازه));
    } وإلا {
        تقسیم_entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(K),
                                                                  ggml_webgpu_tensor_align_جابه_جایی(ctx, K),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, K)));
        تقسیم_entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(2, ggml_webgpu_tensor_buf(V),
                                                                  ggml_webgpu_tensor_align_جابه_جایی(ctx, V),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, V)));
    }
    uint32_t تقسیم_binding_شاخص = op.kv_overlap ? 2u : 3u;
    إذا (op.has_ماسک) {
        تقسیم_entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(تقسیم_binding_شاخص++, ggml_webgpu_tensor_buf(ماسک),
                                                                  ggml_webgpu_tensor_align_جابه_جایی(ctx, ماسک),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, ماسک)));
    }
    إذا (op.has_sinks) {
        تقسیم_entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(تقسیم_binding_شاخص++, ggml_webgpu_tensor_buf(sinks),
                                                                  ggml_webgpu_tensor_align_جابه_جایی(ctx, sinks),
                                                                  ggml_webgpu_tensor_binding_اندازه(ctx, sinks)));
    }
    إذا (op.has_ماسک) {
        تقسیم_entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(تقسیم_binding_شاخص++, blk_buf, blk_entries[1].جابه_جایی, blk_اندازه_بایتها));
    }
    تقسیم_entries.push_back(
        ggml_webgpu_make_bind_گروه_مدخل(تقسیم_binding_شاخص++, tmp_buf, tmp_bind_جابه_جایی, tmp_bind_اندازه));
    تقسیم_entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(تقسیم_binding_شاخص++, ggml_webgpu_tensor_buf(dst),
                                                              ggml_webgpu_tensor_align_جابه_جایی(ctx, dst),
                                                              ggml_webgpu_tensor_binding_اندازه(ctx, dst)));

    webgpu_pipeline                   reduce_pipeline;
    std::vector<uint32_t>             reduce_params;
    std::vector<wgpu::Bind_گروه_مدخل> reduce_entries;
    إذا (use_vec_reduce) {
        ثابت uint32_t reduce_sg_اندازه = ctx->global_ctx->capabilities.max_subgroup_اندازه;
        ثابت uint32_t reduce_wg_اندازه = std::max(
            reduce_sg_اندازه,
            (uint32_t) std::min<uint64_t>((uint64_t) nwg * reduce_sg_اندازه,
                                          ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup));
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
            (uint32_t) tmp_stats_پایه,                                                           // tmp_stats_base
        };

        reduce_entries = {
            ggml_webgpu_make_bind_گروه_مدخل(0, tmp_buf, tmp_bind_جابه_جایی, tmp_اندازه_بایتها),
            ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(dst), ggml_webgpu_tensor_align_جابه_جایی(ctx, dst),
                                              ggml_webgpu_tensor_binding_اندازه(ctx, dst)),
        };
    }

    uint32_t       wg_x           = Q->ne[1] * Q->ne[2] * Q->ne[3];
    ثابت uint64_t تقسیم_wg_مجموع = (uint64_t) wg_x * nwg;
    GGML_ASSERT(تقسیم_wg_مجموع <= UINT32_MAX);

    std::vector<webgpu_dispatch_desc> dispatches;

    إذا (op.has_ماسک) {
        dispatches.push_back({
            blk_pipeline, std::move(blk_params), std::move(blk_entries), { blk_nblk0, blk_nblk1 * blk_batch_شمار }
        });
    }
    dispatches.push_back({
        pipeline, std::move(تقسیم_params), std::move(تقسیم_entries), { (uint32_t) تقسیم_wg_مجموع, 1u }
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
    إذا (ggml_webgpu_flash_attn_use_vec_مسیر(ctx->global_ctx, Q, K, V)) {
        أرجع ggml_webgpu_flash_attn_vec(ctx, Q, K, V, ماسک, sinks, dst, std::move(op));
    }
    أرجع ggml_webgpu_flash_attn_مستقیم(ctx, op);
}

ساكن webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقي is_unary = dst->op == GGML_OP_UNARY;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = اشاره_تهی;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);

    خودکار *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, effective_src),
    };
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_دودویی_op(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
                                               ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    ثابت منطقي      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_دودویی_pipeline(shader_lib_ctx, src_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_دودویی_shader_decisions *>(pipeline.context.get());

    uint32_t ne = (uint32_t) ggml_nelements(dst);

    size_t src0_webgpu_tensor_align_جابه_جایی = ggml_webgpu_tensor_align_جابه_جایی(ctx, src0);
    size_t src1_webgpu_tensor_align_جابه_جایی = ggml_webgpu_tensor_align_جابه_جایی(ctx, src1);

    uint32_t جابه_جایی_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t جابه_جایی_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    size_t   merged_جابه_جایی = 0;
    size_t   merged_اندازه   = 0;
    إذا (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_جابه_جایی = merged_range.جابه_جایی;
        merged_اندازه   = merged_range.size;
        جابه_جایی_src0   = ggml_webgpu_tensor_merged_element_جابه_جایی(src0, merged_range);
        جابه_جایی_src1   = ggml_webgpu_tensor_merged_element_جابه_جایی(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        ne,
        جابه_جایی_src0,
        جابه_جایی_src1,
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

    std::vector<wgpu::Bind_گروه_مدخل> entries;

    إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src0), merged_جابه_جایی, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src0),
                                                            src0_webgpu_tensor_align_جابه_جایی,
                                                            ggml_webgpu_tensor_binding_اندازه(ctx, src0)));
        entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(src1),
                                                            src1_webgpu_tensor_align_جابه_جایی,
                                                            ggml_webgpu_tensor_binding_اندازه(ctx, src1)));
        إذا (!decisions->inplace && !decisions->overlap) {
            entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst));
        }
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_جمع_کن_id(webgpu_context & ctx,
                                            ggml_tensor *    src0,
                                            ggml_tensor *    src1,
                                            ggml_tensor *    src2,
                                            ggml_tensor *    dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.src1                           = src1;
    shader_lib_ctx.src2                           = src2;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_جمع_کن_id_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

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

    std::vector<wgpu::Bind_گروه_مدخل> entries;

    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1));
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, src2));

    إذا (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 3, dst));
    }

    uint32_t       wg_x           = 1;
    uint32_t       wg_y           = 1;
    uint32_t       مجموع_wg       = ggml_nrows(dst);
    ثابت uint32_t max_wg_در_هر_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension;
    compute_2d_workgroups(مجموع_wg, max_wg_در_هر_dim, wg_x, wg_y);

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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    ثابت منطقي      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) ||
                                  ggml_nbytes(src0) == 0 || ggml_nbytes(src1) == 0;
    webgpu_pipeline pipeline    = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx, src_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_دودویی_shader_decisions *>(pipeline.context.get());

    uint32_t جابه_جایی_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t جابه_جایی_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
    size_t   merged_جابه_جایی = 0;
    size_t   merged_اندازه   = 0;
    إذا (decisions->src_overlap) {
        إذا (ggml_nbytes(src0) == 0) {
            merged_جابه_جایی = ggml_webgpu_tensor_align_جابه_جایی(ctx, src1);
            merged_اندازه   = ggml_webgpu_tensor_binding_اندازه(ctx, src1);
            جابه_جایی_src0   = 0;
            جابه_جایی_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type));
        } وإلا إذا (ggml_nbytes(src1) == 0) {
            merged_جابه_جایی = ggml_webgpu_tensor_align_جابه_جایی(ctx, src0);
            merged_اندازه   = ggml_webgpu_tensor_binding_اندازه(ctx, src0);
            جابه_جایی_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
            جابه_جایی_src1   = 0;
        } وإلا {
            ثابت ggml_webgpu_merged_binding_range merged_range =
                ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
            merged_جابه_جایی = merged_range.جابه_جایی;
            merged_اندازه   = merged_range.size;
            جابه_جایی_src0   = ggml_webgpu_tensor_merged_element_جابه_جایی(src0, merged_range);
            جابه_جایی_src1   = ggml_webgpu_tensor_merged_element_جابه_جایی(src1, merged_range);
        }
    }

    std::vector<uint32_t> params = { ne,
                                     جابه_جایی_src0,
                                     جابه_جایی_src1,
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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {};
    إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src0), merged_جابه_جایی, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
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

    std::vector<wgpu::Bind_گروه_مدخل> entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0),
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst),
    };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src0;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ne, decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
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

    uint32_t جابه_جایی_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_اندازه(rn_src->type));
    uint32_t جابه_جایی_mul_src =
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_اندازه(mul_src->type));
    size_t merged_جابه_جایی = 0;
    size_t merged_اندازه   = 0;

    std::vector<uint32_t> params = {
        جابه_جایی_rn_src,
        جابه_جایی_mul_src,
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

    std::vector<wgpu::Bind_گروه_مدخل> entries;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = rn_src;
    shader_lib_ctx.src1                           = mul_src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    ثابت منطقي      src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, rn_src, mul_src);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx, src_overlap);
    خودکار *          decisions   = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());

    إذا (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
        merged_جابه_جایی  = merged_range.جابه_جایی;
        merged_اندازه    = merged_range.size;
        جابه_جایی_rn_src  = ggml_webgpu_tensor_merged_element_جابه_جایی(rn_src, merged_range);
        جابه_جایی_mul_src = ggml_webgpu_tensor_merged_element_جابه_جایی(mul_src, merged_range);
        params[0]      = جابه_جایی_rn_src;
        params[1]      = جابه_جایی_mul_src;
    }

    إذا (decisions->inplace || decisions->overlap) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, mul_src));
    } وإلا إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(rn_src), merged_جابه_جایی, merged_اندازه));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, rn_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, mul_src));
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(dst), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x,
                          wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_ردیف_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_ردیف_norm_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src) };
    إذا (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    compute_2d_workgroups(ggml_nrows(src), ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x,
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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);

    خودکار * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت منطقي inplace         = decisions->inplace;
    ثابت صحيح  has_freq_ضریب = (src2 != اشاره_تهی);

    ثابت صحيح n_dims     = ((int32_t *) dst->op_params)[1];
    ثابت صحيح mode       = ((int32_t *) dst->op_params)[2];
    ثابت صحيح n_ctx_orig = ((int32_t *) dst->op_params)[4];
    ثابت صحيح n_offs     = ((int32_t *) dst->op_params)[15];

    عائم freq_پایه;
    عائم freq_scale;
    عائم ext_ضریب;
    عائم attn_ضریب;
    عائم beta_fast;
    عائم beta_slow;
    کپی_حافظه(&freq_پایه, (int32_t *) dst->op_params + 5, حجم(عائم));
    کپی_حافظه(&freq_scale, (int32_t *) dst->op_params + 6, حجم(عائم));
    کپی_حافظه(&ext_ضریب, (int32_t *) dst->op_params + 7, حجم(عائم));
    کپی_حافظه(&attn_ضریب, (int32_t *) dst->op_params + 8, حجم(عائم));
    کپی_حافظه(&beta_fast, (int32_t *) dst->op_params + 9, حجم(عائم));
    کپی_حافظه(&beta_slow, (int32_t *) dst->op_params + 10, حجم(عائم));

    صحيح sections[4];
    کپی_حافظه(sections, (int32_t *) dst->op_params + 11, 4 * حجم(صحيح));

    عائم theta_scale = powf(freq_پایه, -2.0f / n_dims);

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_پایه, beta_fast, beta_slow, corr_dims);

    std::vector<uint32_t> params = {
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type)),
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)),
        src2 != اشاره_تهی ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_اندازه(src2->type)) : 0,
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

    std::vector<wgpu::Bind_گروه_مدخل> entries     = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0),
                                                      ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1) };
    uint32_t                          dst_binding = 2;
    إذا (has_freq_ضریب) {
        dst_binding = 3;
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 2, src2));
    }
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, dst_binding, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    ثابت منطقي      src_overlap = src1 != اشاره_تهی && ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1);
    webgpu_pipeline pipeline    = ctx->shader_lib->get_glu_pipeline(shader_lib_ctx, src_overlap);

    خودکار * decisions = static_cast<ggml_webgpu_glu_shader_decisions *>(pipeline.context.get());

    ثابت صحيح تقسیم = (src1 != اشاره_تهی);

    uint32_t جابه_جایی_src0 = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_اندازه(src0->type));
    uint32_t جابه_جایی_src1 =
        src1 != اشاره_تهی ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_اندازه(src1->type)) : 0;
    size_t merged_جابه_جایی = 0;
    size_t merged_اندازه   = 0;
    إذا (decisions->src_overlap) {
        ثابت ggml_webgpu_merged_binding_range merged_range =
            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
        merged_جابه_جایی = merged_range.جابه_جایی;
        merged_اندازه   = merged_range.size;
        جابه_جایی_src0   = ggml_webgpu_tensor_merged_element_جابه_جایی(src0, merged_range);
        جابه_جایی_src1   = ggml_webgpu_tensor_merged_element_جابه_جایی(src1, merged_range);
    }

    std::vector<uint32_t> params = {
        جابه_جایی_src0,
        جابه_جایی_src1,
        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
        (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        (uint32_t) (src0->nb[3] / ggml_type_اندازه(src0->type)),
        src1 != اشاره_تهی ? (uint32_t) (src1->nb[1] / ggml_type_اندازه(src1->type)) :
                          (uint32_t) (src0->nb[1] / ggml_type_اندازه(src0->type)),
        src1 != اشاره_تهی ? (uint32_t) (src1->nb[2] / ggml_type_اندازه(src1->type)) :
                          (uint32_t) (src0->nb[2] / ggml_type_اندازه(src0->type)),
        src1 != اشاره_تهی ? (uint32_t) (src1->nb[3] / ggml_type_اندازه(src1->type)) :
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

    std::vector<wgpu::Bind_گروه_مدخل> entries;
    uint32_t                          dst_binding = 1;
    إذا (decisions->src_overlap) {
        entries.push_back(
            ggml_webgpu_make_bind_گروه_مدخل(0, ggml_webgpu_tensor_buf(src0), merged_جابه_جایی, merged_اندازه));
    } وإلا {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src0));
    }
    إذا (تقسیم && !decisions->src_overlap) {
        dst_binding = 2;
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, src1));
    }
    entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, dst_binding, dst));

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = اشاره_تهی;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

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
    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src) };

    إذا (!decisions->inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV(ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
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
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    ثابت منطقي inplace     = decisions->inplace;
    ثابت صحيح  has_ماسک    = (src1 != اشاره_تهی);
    ثابت صحيح  has_sink    = (src2 != اشاره_تهی);
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

    std::vector<wgpu::Bind_گروه_مدخل> entries     = { ggml_webgpu_make_bind_گروه_مدخل(
        0, ggml_webgpu_tensor_buf(src0), ggml_webgpu_tensor_align_جابه_جایی(ctx, src0),
        ggml_webgpu_tensor_binding_اندازه(ctx, src0)) };
    uint32_t                          binding_رقم = 1;
    إذا (has_ماسک) {
        entries.push_back(ggml_webgpu_make_bind_گروه_مدخل(binding_رقم, ggml_webgpu_tensor_buf(src1),
                                                            ggml_webgpu_tensor_align_جابه_جایی(ctx, src1),
                                                            ggml_webgpu_tensor_binding_اندازه(ctx, src1)));
        binding_رقم++;
    }
    إذا (has_sink) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم, src2));
        binding_رقم++;
    }
    إذا (!inplace) {
        entries.push_back(ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, binding_رقم, dst));
    }

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = ggml_nrows(dst);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
    uint32_t        wg_x     = ggml_nelements(dst);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
}

ساكن webgpu_encoded_op ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقي is_بالاترین_k = dst->op == GGML_OP_بالاترین_K;

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = اشاره_تهی;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه        = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
    shader_lib_ctx.wg_mem_حد_بایتها = ctx->global_ctx->capabilities.limits.max_Compute_Workgroup_Storage_اندازه;

    webgpu_pipeline argsort_pipeline = ctx->shader_lib->get_argsort_pipeline(shader_lib_ctx);
    خودکار * argsort_decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(argsort_pipeline.context.get());

    webgpu_pipeline argsort_ادغام_pipeline = ctx->shader_lib->get_argsort_ادغام_pipeline(shader_lib_ctx);

    ثابت uint32_t src_ne0 = (uint32_t) src->ne[0];
    ثابت uint32_t nrows   = (uint32_t) ggml_nrows(src);
    ثابت uint32_t npr     = CEIL_DIV(src_ne0, argsort_decisions->wg_اندازه);
    ثابت uint32_t بلوک_اندازه_3 =
        is_بالاترین_k ? std::min(argsort_decisions->wg_اندازه, (uint32_t) dst->ne[0]) : argsort_decisions->wg_اندازه;
    uint32_t بیرون_ne0 = src_ne0;
    إذا (is_بالاترین_k) {
        إذا (npr > 1) {
            ثابت uint32_t آخرین_tile = src_ne0 - (npr - 1) * argsort_decisions->wg_اندازه;
            بیرون_ne0                  = (npr - 1) * بلوک_اندازه_3 + std::min(آخرین_tile, بلوک_اندازه_3);
        } وإلا {
            بیرون_ne0 = بلوک_اندازه_3;
        }
    }

    uint32_t ادغام_len    = بلوک_اندازه_3;
    uint32_t ادغام_passes = 0;
    طالما (ادغام_len < بیرون_ne0) {
        ادغام_len <<= 1;
        ادغام_passes++;
    }

    ثابت منطقي شروع_درون_tmp = (ادغام_passes % 2) == 1;

    ثابت size_t dst_جابه_جایی = ggml_webgpu_tensor_جابه_جایی(dst);
    ثابت size_t idx_nbytes = بیرون_ne0 * ggml_nrows(dst) * حجم(int32_t);
    ثابت size_t tmp_جابه_جایی =
        ROUNDUP_POW2(dst_جابه_جایی + idx_nbytes, ctx->global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment);
    ثابت size_t tmp_binding_اندازه = ROUNDUP_POW2(idx_nbytes, WEBGPU_STORAGE_BUF_BINDING_MULT);
    ثابت size_t dst_binding_اندازه =
        ROUNDUP_POW2(idx_nbytes + ggml_webgpu_tensor_misalignment(ctx, dst), WEBGPU_STORAGE_BUF_BINDING_MULT);

    ثابت uint32_t جابه_جایی_src  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type));
    ثابت uint32_t جابه_جایی_dst  = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type));
    ثابت uint32_t جابه_جایی_tmp  = 0;
    ثابت uint32_t stride_src1 = (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src2 = (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_src3 = (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type));
    ثابت uint32_t stride_idx1 = بیرون_ne0;
    ثابت uint32_t stride_idx2 = بیرون_ne0 * (uint32_t) dst->ne[1];
    ثابت uint32_t stride_idx3 = stride_idx2 * (uint32_t) dst->ne[2];

    std::vector<webgpu_dispatch_desc> dispatches;

    ثابت uint32_t init_جابه_جایی       = شروع_درون_tmp ? جابه_جایی_tmp : جابه_جایی_dst;
    ثابت size_t   init_align_جابه_جایی = شروع_درون_tmp ? tmp_جابه_جایی : ggml_webgpu_tensor_align_جابه_جایی(ctx, dst);
    ثابت size_t   init_binding_اندازه = شروع_درون_tmp ? tmp_binding_اندازه : dst_binding_اندازه;

    std::vector<uint32_t> init_params = {
        جابه_جایی_src,  init_جابه_جایی, stride_src1, stride_src2,           stride_src3,           stride_idx1,
        stride_idx2, stride_idx3, src_ne0,     (uint32_t) src->ne[1], (uint32_t) src->ne[2], بیرون_ne0,
        بلوک_اندازه_3,  npr,         nrows
    };

    uint32_t       wg_x_init;
    uint32_t       wg_y_init;
    ثابت uint32_t مجموع_wg_init  = npr * nrows;
    ثابت uint32_t max_wg_در_هر_dim = ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension;
    compute_2d_workgroups(مجموع_wg_init, max_wg_در_هر_dim, wg_x_init, wg_y_init);

    std::vector<wgpu::Bind_گروه_مدخل> init_entries = {
        ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
        ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(dst), init_align_جابه_جایی, init_binding_اندازه)
    };

    dispatches.push_back({
        argsort_pipeline, std::move(init_params), std::move(init_entries), { wg_x_init, wg_y_init }
    });

    إذا (ادغام_passes == 0) {
        أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
    }

    منطقي     درون_is_tmp = شروع_درون_tmp;
    uint32_t len       = بلوک_اندازه_3;
    طالما (len < بیرون_ne0) {
        ثابت uint32_t nm = CEIL_DIV(بیرون_ne0, 2 * len);

        ثابت منطقي     بیرون_is_tmp  = !درون_is_tmp;
        ثابت uint32_t جابه_جایی_درون   = درون_is_tmp ? جابه_جایی_tmp : جابه_جایی_dst;
        ثابت uint32_t جابه_جایی_بیرون  = بیرون_is_tmp ? جابه_جایی_tmp : جابه_جایی_dst;
        ثابت size_t   align_درون    = درون_is_tmp ? tmp_جابه_جایی : ggml_webgpu_tensor_align_جابه_جایی(ctx, dst);
        ثابت size_t   align_بیرون   = بیرون_is_tmp ? tmp_جابه_جایی : ggml_webgpu_tensor_align_جابه_جایی(ctx, dst);
        ثابت size_t   اندازه_درون     = درون_is_tmp ? tmp_binding_اندازه : dst_binding_اندازه;
        ثابت size_t   اندازه_بیرون    = بیرون_is_tmp ? tmp_binding_اندازه : dst_binding_اندازه;
        ثابت uint32_t بالاترین_k_بیرون   = (is_بالاترین_k && nm == 1) ? (uint32_t) dst->ne[0] : بیرون_ne0;
        ثابت uint32_t stride_out1 = بالاترین_k_بیرون;
        ثابت uint32_t stride_out2 = بالاترین_k_بیرون * (uint32_t) dst->ne[1];
        ثابت uint32_t stride_out3 = stride_out2 * (uint32_t) dst->ne[2];

        std::vector<uint32_t> ادغام_params = { جابه_جایی_src,
                                               جابه_جایی_درون,
                                               جابه_جایی_بیرون,
                                               stride_src1,
                                               stride_src2,
                                               stride_src3,
                                               stride_idx1,
                                               stride_idx2,
                                               stride_idx3,
                                               stride_out1,
                                               stride_out2,
                                               stride_out3,
                                               بیرون_ne0,
                                               (uint32_t) src->ne[1],
                                               (uint32_t) src->ne[2],
                                               بالاترین_k_بیرون,
                                               len,
                                               nm,
                                               nrows };

        std::vector<wgpu::Bind_گروه_مدخل> ادغام_entries = {
            ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
            ggml_webgpu_make_bind_گروه_مدخل(1, ggml_webgpu_tensor_buf(dst), align_درون, اندازه_درون),
            ggml_webgpu_make_bind_گروه_مدخل(2, ggml_webgpu_tensor_buf(dst), align_بیرون, اندازه_بیرون)
        };

        uint32_t       wg_x_ادغام;
        uint32_t       wg_y_ادغام;
        ثابت uint32_t مجموع_wg_ادغام = nm * nrows;
        compute_2d_workgroups(مجموع_wg_ادغام, max_wg_در_هر_dim, wg_x_ادغام, wg_y_ادغام);

        dispatches.push_back({
            argsort_ادغام_pipeline, std::move(ادغام_params), std::move(ادغام_entries), { wg_x_ادغام, wg_y_ادغام }
        });

        len <<= 1;
        درون_is_tmp = !درون_is_tmp;
    }

    أرجع ggml_backend_webgpu_build_multi(ctx, dispatches);
}

ساكن webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     (uint32_t) src->ne[0] };

    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.src1                           = اشاره_تهی;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
    uint32_t        wg_x;
    uint32_t        wg_y;
    uint32_t        مجموع_wg = ggml_nrows(dst);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن webgpu_encoded_op ggml_webgpu_جمع_ردیفها(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
    منطقي                  مجموع_جمع = dst->op == GGML_OP_جمع;
    std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_اندازه(src->type)),
                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_اندازه(dst->type)),
                                     مجموع_جمع ? 0 : (uint32_t) (src->nb[1] / ggml_type_اندازه(src->type)),
                                     مجموع_جمع ? 0 : (uint32_t) (src->nb[2] / ggml_type_اندازه(src->type)),
                                     مجموع_جمع ? 0 : (uint32_t) (src->nb[3] / ggml_type_اندازه(src->type)),
                                     مجموع_جمع ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                     مجموع_جمع ? 1 : (uint32_t) src->ne[1],
                                     مجموع_جمع ? 1 : (uint32_t) src->ne[2] };

    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline = ctx->shader_lib->get_جمع_ردیفها_pipeline(shader_lib_ctx);

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = مجموع_جمع ? 1 : ggml_nrows(dst);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);
    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

ساكن منطقي ggml_webgpu_can_fuse_rms_norm_mul(ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx) {
    إذا (!ggml_can_fuse(cgraph, گره_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        أرجع خطأ;
    }

    // additional constraints specific to this fusion
    ثابت ggml_tensor * rms_norm = cgraph->nodes[گره_idx];
    ثابت ggml_tensor * mul      = cgraph->nodes[گره_idx + 1];

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
    إذا (!ggml_is_contiguous_ردیفها(mul->src[0]) || !ggml_is_contiguous_ردیفها(mul->src[1])) {
        أرجع خطأ;
    }

    أرجع درست;
}

ساكن منطقي ggml_webgpu_can_fuse_gdn_cache(ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, صحيح & رقم_encoded_ops) {
    ثابت ggml_tensor * gdn = cgraph->nodes[گره_idx];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    إذا (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * src_v     = gdn->src[2];
    ثابت int64_t       S_v       = src_v->ne[0];
    ثابت int64_t       H         = src_v->ne[1];
    ثابت int64_t       n_نشانهها  = src_v->ne[2];
    ثابت int64_t       n_seqs    = src_v->ne[3];
    ثابت int64_t       D         = S_v * S_v * H;
    ثابت int64_t       K         = ggml_get_op_params_i32(gdn, 0);  // snapshot slot count
    ثابت int64_t       n_written = std::min<int64_t>(n_نشانهها, K);  // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ثابت size_t دم_off = ggml_ردیف_اندازه(GGML_TYPE_F32, S_v * H * n_نشانهها * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    ثابت ggml_tensor * cpy     = اشاره_تهی;
    صحيح                 cpy_idx = 0;
    لكل (صحيح j = گره_idx + 1; j < cgraph->n_nodes && cpy == اشاره_تهی; ++j) {
        ثابت ggml_tensor * n = cgraph->nodes[j];
        إذا (ggml_op_is_خالی(n->op) || ggml_is_خالی(n)) {
            استمر;
        }
        إذا (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            أرجع خطأ;
        }
        cpy     = n;
        cpy_idx = j;
    }
    إذا (cpy == اشاره_تهی) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * cpy_src = cpy->src[0];  // view of the gdn snapshot tail
    ثابت ggml_tensor * cpy_dst = cpy->src[1];  // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    إذا (cpy_src->op != GGML_OP_VIEW || cpy_src->view_src != gdn || cpy_src->view_offs != دم_off ||
        !ggml_is_contiguous(cpy_src)) {
        أرجع خطأ;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    ثابت std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    إذا (cpy_dst->op != GGML_OP_VIEW || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->داده == اشاره_تهی ||
        !std::equal(expected_ne.begin(), expected_ne.end(), cpy_dst->ne) ||
        cpy_dst->nb[0] != ggml_type_اندازه(GGML_TYPE_F32) || cpy_dst->nb[1] != (size_t) ggml_ردیف_اندازه(GGML_TYPE_F32, D)) {
        أرجع خطأ;
    }

    رقم_encoded_ops = cpy_idx - گره_idx + 1;

    أرجع درست;
}

ساكن webgpu_encoded_op ggml_webgpu_upscale(webgpu_context ctx, ggml_tensor * src, ggml_tensor * dst) {
    ثابت uint32_t        نما_flags = (uint32_t) ggml_get_op_params_i32(dst, 0);
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

                                     نما_flags };

    std::vector<wgpu::Bind_گروه_مدخل> entries = { ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 0, src),
                                                  ggml_webgpu_make_tensor_bind_گروه_مدخل(ctx, 1, dst) };

    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
    shader_lib_ctx.src0                           = src;
    shader_lib_ctx.dst                            = dst;
    shader_lib_ctx.max_wg_اندازه = ctx->global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;

    webgpu_pipeline pipeline  = ctx->shader_lib->get_upscale_pipeline(shader_lib_ctx);
    خودکار *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());

    uint32_t wg_x;
    uint32_t wg_y;
    uint32_t مجموع_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_اندازه);
    compute_2d_workgroups(مجموع_wg, ctx->global_ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension, wg_x, wg_y);

    أرجع ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
}

// Returns the encoded command, or std::nullopt if the operation is a no-op
ساكن std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
                                                           ggml_cgraph *  cgraph,
                                                           صحيح            گره_idx,
                                                           صحيح &          رقم_encoded_ops) {
    ggml_tensor ** nodes = cgraph->nodes;
    ggml_tensor *  گره  = nodes[گره_idx];

    إذا (ggml_is_خالی(گره)) {
        أرجع std::nullopt;
    }
    إذا ((گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        أرجع std::nullopt;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << گره << ", " << ggml_op_name(گره->op) << ")");

    ggml_tensor * src0 = گره->src[0];
    ggml_tensor * src1 = گره->src[1];
    ggml_tensor * src2 = گره->src[2];

    بدل (گره->op) {
            // no-ops
        حالة GGML_OP_NONE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_جایگشت:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE:
            أرجع std::nullopt;
        حالة GGML_OP_DUP:
        حالة GGML_OP_CPY:
        حالة GGML_OP_CONT:
            أرجع ggml_webgpu_cpy(ctx, src0, گره);
        حالة GGML_OP_مجموعه:
            أرجع ggml_webgpu_مجموعه(ctx, src0, src1, گره);
        حالة GGML_OP_مجموعه_ردیفها:
            أرجع ggml_webgpu_مجموعه_ردیفها(ctx, src0, src1, گره);
        حالة GGML_OP_GET_ردیفها:
            أرجع ggml_webgpu_get_ردیفها(ctx, src0, src1, گره);
        حالة GGML_OP_MUL_MAT:
            أرجع ggml_webgpu_mul_mat(ctx, src0, src1, گره);
        حالة GGML_OP_MUL_MAT_ID:
            أرجع ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, گره);
        حالة GGML_OP_FLASH_ATTN_EXT:
            أرجع ggml_webgpu_flash_attn(ctx, src0, src1, src2, گره->src[3], گره->src[4], گره);
        حالة GGML_OP_جمع_کن:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
            أرجع ggml_webgpu_دودویی_op(ctx, src0, src1, گره);
        حالة GGML_OP_جمع_کن_ID:
            أرجع ggml_webgpu_جمع_کن_id(ctx, src0, src1, src2, گره);
        حالة GGML_OP_CONCAT:
            أرجع ggml_webgpu_concat(ctx, src0, src1, گره);
        حالة GGML_OP_REPEAT:
            أرجع ggml_webgpu_repeat(ctx, src0, گره);
        حالة GGML_OP_RMS_NORM:
            إذا (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, گره_idx)) {
                رقم_encoded_ops        = 2;
                ggml_tensor * mul_گره = nodes[گره_idx + 1];
                أرجع ggml_webgpu_rms_norm_mul(ctx, src0, گره, mul_گره->src[0], mul_گره->src[1], mul_گره);
            } وإلا {
                أرجع ggml_webgpu_ردیف_norm(ctx, src0, گره);
            }
        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
            أرجع ggml_webgpu_ردیف_norm(ctx, src0, گره);
        حالة GGML_OP_ROPE:
            أرجع ggml_webgpu_rope(ctx, src0, src1, src2, گره);
        حالة GGML_OP_GLU:
            أرجع ggml_webgpu_glu(ctx, src0, src1, گره);
        حالة GGML_OP_SCALE:
            أرجع ggml_webgpu_scale(ctx, src0, گره);
        حالة GGML_OP_SOFT_MAX:
            أرجع ggml_webgpu_soft_max(ctx, src0, src1, src2, گره);
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
            أرجع ggml_webgpu_unary_op(ctx, src0, گره);
        حالة GGML_OP_SOLVE_TRI:
            أرجع ggml_webgpu_solve_tri(ctx, src0, src1, گره);
        حالة GGML_OP_SSM_CONV:
            أرجع ggml_webgpu_ssm_conv(ctx, src0, src1, گره);
        حالة GGML_OP_SSM_پیمایش:
            أرجع ggml_webgpu_ssm_پیمایش(ctx, src0, src1, src2, گره->src[3], گره->src[4], گره->src[5], گره->src[6],
                                        گره);
        حالة GGML_OP_GATED_DELTA_NET:
            إذا (ggml_webgpu_can_fuse_gdn_cache(cgraph, گره_idx, رقم_encoded_ops)) {
                ggml_tensor * dst_fuse = cgraph->nodes[گره_idx + رقم_encoded_ops - 1]->src[1];
                أرجع ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, گره->src[3], گره->src[4], گره->src[5],
                                                   گره, dst_fuse);
            } وإلا {
                أرجع ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, گره->src[3], گره->src[4], گره->src[5],
                                                   گره, اشاره_تهی);
            }
        حالة GGML_OP_PAD:
            أرجع ggml_webgpu_pad(ctx, src0, گره);
        حالة GGML_OP_ARGMAX:
            أرجع ggml_webgpu_argmax(ctx, src0, گره);
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_بالاترین_K:
            // we reuse the same argsort implementation for top_k
            أرجع ggml_webgpu_argsort(ctx, src0, گره);
        حالة GGML_OP_CUMSUM:
            أرجع ggml_webgpu_cumsum(ctx, src0, گره);
        حالة GGML_OP_جمع:
        حالة GGML_OP_جمع_ردیفها:
            أرجع ggml_webgpu_جمع_ردیفها(ctx, src0, گره);
        حالة GGML_OP_CONV_2D:
            أرجع ggml_webgpu_conv_2d(ctx, src0, src1, گره);
        حالة GGML_OP_CONV_2D_DW:
            أرجع ggml_webgpu_conv_2d_dw(ctx, src0, src1, گره);
        حالة GGML_OP_IM2_ستون:
            أرجع ggml_webgpu_im2col(ctx, src0, src1, گره);
        حالة GGML_OP_UPSCALE:
            أرجع ggml_webgpu_upscale(ctx, src0, گره);
        افتراضي:
            أرجع std::nullopt;
    }
}

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
ساكن فراغ ggml_backend_webgpu_collect_profile_results(webgpu_context &                 ctx,
                                                        ثابت std::vector<std::string> & pipeline_نامها,
                                                        uint32_t &                       رقم_inflight_batches) {
    إذا (pipeline_نامها.خالی()) {
        أرجع;
    }

    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Resolve_Query_مجموعه(ctx->profile_timestamp_query_مجموعه, 0, ctx->profile_timestamp_query_شمار,
                            ctx->profile_timestamp_dev_buf, 0);
    encoder.Copy_میانگیر_مقصد_میانگیر(ctx->profile_timestamp_dev_buf, 0, ctx->profile_timestamp_host_buf, 0,
                               ctx->profile_timestamp_query_شمار * حجم(uint64_t));

    wgpu::Command_میانگیر profile_commands = encoder.پایان();
    ggml_backend_webgpu_submit_commands(ctx, profile_commands, رقم_inflight_batches);

    ثابت size_t mapped_اندازه = ctx->profile_timestamp_query_شمار * حجم(uint64_t);
    GGML_ASSERT(ctx->profile_timestamp_query_شمار == 2 * pipeline_نامها.size());

    ggml_backend_webgpu_map_میانگیر(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::Map_نما::Read, 0,
                                   mapped_اندازه);
    ثابت uint64_t * ts_داده = (ثابت uint64_t *) ctx->profile_timestamp_host_buf.GetConstMappedRange(0, mapped_اندازه);

    لكل (size_t i = 0; i < pipeline_نامها.size(); ++i) {
        // WebGPU timestamps are in ns; convert to ms.
        ثابت مزدوج elapsed_ms = مزدوج(ts_داده[2 * i + 1] - ts_داده[2 * i]) * 1e-6;
        ctx->shader_gpu_زمان_ms[pipeline_نامها[i]] += elapsed_ms;
    }

    ctx->profile_timestamp_host_buf.Unmap();
}
#پایان_اگر

// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
// models that would require it right now.
ساكن فراغ ggml_backend_webgpu_check_مجموعه_ردیفها(webgpu_context & ctx, uint32_t & رقم_inflight_batches) {
#اگر_تعریف_شده GGML_WEBGPU_CHECK_مجموعه_ردیفها
    wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
    encoder.Copy_میانگیر_مقصد_میانگیر(ctx->مجموعه_ردیفها_dev_error_buf, 0, ctx->مجموعه_ردیفها_host_error_buf, 0,
                               ctx->مجموعه_ردیفها_host_error_buf.Get_اندازه());
    wgpu::Command_میانگیر commands = encoder.پایان();
    ggml_backend_webgpu_submit_commands(ctx, commands, رقم_inflight_batches);
    ggml_backend_webgpu_map_میانگیر(ctx->global_ctx, ctx->مجموعه_ردیفها_host_error_buf, wgpu::Map_نما::Read, 0,
                                   ctx->مجموعه_ردیفها_host_error_buf.Get_اندازه());
    ثابت uint32_t * error_داده = (ثابت uint32_t *) ctx->مجموعه_ردیفها_host_error_buf.GetConstMappedRange();
    إذا (*error_داده) {
        GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
    }
    ctx->مجموعه_ردیفها_host_error_buf.Unmap();
#وإلا
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(رقم_inflight_batches);
#پایان_اگر
}

ساكن ggml_وضعیت ggml_backend_webgpu_گراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_graph_compute(" << cgraph->n_nodes << " nodes)");

    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    webgpu_context                ctx         = backend_ctx->webgpu_ctx;

    WEBGPU_CPU_PROFILE_مجموع_شروع(گراف_compute);

    std::vector<webgpu_encoded_op> commands;

    uint32_t رقم_batched_kernels  = 0;
    uint32_t رقم_inflight_batches = 0;
    منطقي     شامل_است_مجموعه_ردیفها    = خطأ;
    صحيح      رقم_encoded_ops      = 1;
    صحيح      گره_idx             = 0;

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    ctx->profile_timestamp_query_شمار = 0;
    std::vector<std::string> profile_pipeline_نامها;
#پایان_اگر

    ctx->فعال_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
    إذا (ctx->batch_compute_passes) {
        ctx->فعال_compute_pass = ctx->فعال_command_encoder.BeginComputePass();
    }

    طالما (گره_idx < cgraph->n_nodes) {
        إذا (cgraph->nodes[گره_idx]->op == GGML_OP_مجموعه_ردیفها) {
            شامل_است_مجموعه_ردیفها = درست;
        }
        إذا (خودکار cmd = ggml_webgpu_encode(ctx, cgraph, گره_idx, رقم_encoded_ops)) {
            commands.push_back(*cmd);
            رقم_batched_kernels += cmd.value().رقم_kernels;
#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
            profile_pipeline_نامها.insert(profile_pipeline_نامها.end(), cmd->pipeline_نامها.begin(),
                                          cmd->pipeline_نامها.end());
#پایان_اگر
        }

        إذا (رقم_batched_kernels >= ctx->global_ctx->command_submit_batch_اندازه) {
            إذا (ctx->فعال_compute_pass) {
                ctx->فعال_compute_pass.انتها_2();
            }
            رقم_batched_kernels                = 0;
            wgpu::Command_میانگیر batch_commands = ctx->فعال_command_encoder.پایان();
            ggml_backend_webgpu_submit_commands(ctx, batch_commands, رقم_inflight_batches);

            // reset state for next batch
            ctx->فعال_command_encoder = ctx->global_ctx->device.CreateCommandEncoder();
            إذا (ctx->batch_compute_passes) {
                ctx->فعال_compute_pass = ctx->فعال_command_encoder.BeginComputePass();
            }
            ctx->param_arena.reset();
            commands.clear();
#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
            // flush before the next batch can overflow the QuerySet
            إذا (ctx->profile_timestamp_query_شمار + 2 * ctx->global_ctx->command_submit_batch_اندازه >=
                WEBGPU_MAX_PROFILE_QUERY_شمار) {
                ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_نامها, رقم_inflight_batches);
                // reset profile timestamp state
                ctx->profile_timestamp_query_شمار = 0;
                profile_pipeline_نامها.clear();
            }
#پایان_اگر
        }

        گره_idx += رقم_encoded_ops;
        رقم_encoded_ops = 1;
    }

    إذا (ctx->فعال_compute_pass) {
        ctx->فعال_compute_pass.انتها_2();
        ctx->فعال_compute_pass = اشاره_تهی;
    }

    إذا (رقم_batched_kernels > 0) {
        wgpu::Command_میانگیر batch_commands = ctx->فعال_command_encoder.پایان();
        ggml_backend_webgpu_submit_commands(ctx, batch_commands, رقم_inflight_batches);
        ctx->param_arena.reset();
        commands.clear();
    }
    ctx->فعال_command_encoder = اشاره_تهی;

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    ggml_backend_webgpu_collect_profile_results(ctx, profile_pipeline_نامها, رقم_inflight_batches);
#پایان_اگر

    إذا (شامل_است_مجموعه_ردیفها) {
        ggml_backend_webgpu_check_مجموعه_ردیفها(ctx, رقم_inflight_batches);
    }

    WEBGPU_CPU_PROFILE_مجموع_انتها(گراف_compute, ctx->global_ctx);
    أرجع GGML_وضعیت_SUCCESS;
}

بنية ggml_backend_webgpu_event_context {
    webgpu_global_context global_ctx;
    wgpu::Future          future;
    منطقي                  recorded = خطأ;
};

ساكن ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;

    خودکار * event_ctx      = جديد ggml_backend_webgpu_event_context();
    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;

    خودکار * event   = جديد ggml_backend_event;
    event->device  = device;
    event->context = event_ctx;
    أرجع event;
}

ساكن فراغ ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_استفادهنشده(dev);
    حذف static_cast<ggml_backend_webgpu_event_context *>(event->context);
    حذف event;
}

ساكن فراغ ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_استفادهنشده(dev);
    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
    إذا (!event_ctx->recorded) {
        أرجع;
    }
    wgpu::Wait_وضعیت وضعیت =
        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    إذا (وضعیت == wgpu::Wait_وضعیت::Timed_بیرون) {
        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
    }
    event_ctx->recorded = خطأ;
}

ساكن فراغ ggml_backend_webgpu_event_رکورد(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;

    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.On_Submitted_Work_انجامشده(
        wgpu::Callback_نما::AllowSpontaneous, [](wgpu::صف_Work_انجامشده_وضعیت, wgpu::StringView) {});
    event_ctx->recorded = درست;
}

ساكن فراغ ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_استفادهنشده(backend);
    ggml_backend_webgpu_device_event_synchronize(اشاره_تهی, event);
}

ساكن فراغ ggml_backend_webgpu_مجموعه_tensor_async(ggml_backend_t backend,
                                                 ggml_tensor *  tensor,
                                                 ثابت فراغ *   داده,
                                                 size_t         جابه_جایی,
                                                 size_t         size) {
    GGML_استفادهنشده(backend);
    خودکار * buf_ctx      = (ggml_backend_webgpu_میانگیر_context *) tensor->میانگیر->context;
    size_t مجموع_جابه_جایی = ggml_webgpu_tensor_جابه_جایی(tensor) + جابه_جایی;

    // Write aligned portion
    buf_ctx->global_ctx->queue.Write_میانگیر(buf_ctx->میانگیر, مجموع_جابه_جایی, داده, (size / 4) * 4);

    إذا (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t مانده_اندازه = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        لكل (size_t i = 0; i < مانده_اندازه; i++) {
            ((uint8_t *) &val32)[i] = ((ثابت uint8_t *) داده)[size - مانده_اندازه + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_میانگیر_memset(buf_ctx->global_ctx, buf_ctx->میانگیر, val32,
                                          مجموع_جابه_جایی + (size - مانده_اندازه), مانده_اندازه);
    }
}

ساكن فراغ ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
    ggml_backend_webgpu_wait_صف(backend_ctx->webgpu_ctx->global_ctx);
}

ساكن ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ ggml_backend_webgpu_مجموعه_tensor_async,
    /* .get_tensor_async        = */ عدم,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_webgpu_گراف_compute,
    /* .event_record            = */ ggml_backend_webgpu_event_رکورد,
    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
    /* .graph_optimize          = */ عدم,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

ساكن فراغ ggml_backend_webgpu_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    ggml_backend_webgpu_میانگیر_context * ctx = static_cast<ggml_backend_webgpu_میانگیر_context *>(میانگیر->context);
    إذا (ctx != اشاره_تهی && ctx->میانگیر != اشاره_تهی) {
        ctx->میانگیر.Destroy();
        حذف ctx;
    }
}

// Returns the "fake" base pointer.
ساكن فراغ * ggml_backend_webgpu_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر) {
    GGML_استفادهنشده(میانگیر);
    أرجع webgpu_ptr_پایه;
}

ساكن فراغ ggml_backend_webgpu_میانگیر_memset_tensor(ggml_backend_میانگیر_t میانگیر,
                                                     ggml_tensor *         tensor,
                                                     uint8_t               value,
                                                     size_t                جابه_جایی,
                                                     size_t                size) {
    إذا (size == 0) {
        WEBGPU_LOG_DEBUG(
            "ggml_backend_webgpu_buffer_memset_tensor: size is zero, "
            "nothing to do.");
        أرجع;
    }

    WEBGPU_CPU_PROFILE_مجموع_شروع(memset_tensor);

    ggml_backend_webgpu_میانگیر_context * buf_ctx = (ggml_backend_webgpu_میانگیر_context *) میانگیر->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                 << ", " << جابه_جایی << ", " << size << ")");

    size_t مجموع_جابه_جایی = ggml_webgpu_tensor_جابه_جایی(tensor) + جابه_جایی;

    // This is a trick to set all bytes of a u32 to the same 1 byte value.
    uint32_t val32 = (uint32_t) value * 0x01010101;
    ggml_backend_webgpu_میانگیر_memset(buf_ctx->global_ctx, buf_ctx->میانگیر, val32, مجموع_جابه_جایی, size);
    WEBGPU_CPU_PROFILE_مجموع_انتها(memset_tensor, buf_ctx->global_ctx);
}

ساكن فراغ ggml_backend_webgpu_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر,
                                                  ggml_tensor *         tensor,
                                                  ثابت فراغ *          داده,
                                                  size_t                جابه_جایی,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_مجموع_شروع(مجموعه_tensor);
    ggml_backend_webgpu_میانگیر_context * buf_ctx = (ggml_backend_webgpu_میانگیر_context *) میانگیر->context;

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << داده
                                                              << ", " << جابه_جایی << ", " << size << ")");

    size_t مجموع_جابه_جایی = ggml_webgpu_tensor_جابه_جایی(tensor) + جابه_جایی;

    // WriteBuffer needs the offset and the size to be multiples of 4.
    // Write the misaligned head bytes using compute memset, then increment total_offset
    // and data pointer so that they are 4-aligned.
    إذا (مجموع_جابه_جایی % 4 != 0) {
        size_t lane = مجموع_جابه_جایی % 4;  // in-word lane the head starts at (the tail below always starts at 0)
        size_t سر = std::min<size_t>(4 - lane, size);

        // Pack head bytes into a uint32_t
        uint32_t سر_val = 0;
        لكل (size_t i = 0; i < سر; i++) {
            ((uint8_t *) &سر_val)[lane + i] = ((ثابت uint8_t *) داده)[i];
        }
        ggml_backend_webgpu_میانگیر_memset(buf_ctx->global_ctx, buf_ctx->میانگیر, سر_val, مجموع_جابه_جایی, سر);

        مجموع_جابه_جایی += سر;
        size -= سر;
        داده = (ثابت uint8_t *) داده + سر;
    }

    إذا (size > 0) {
        buf_ctx->global_ctx->queue.Write_میانگیر(buf_ctx->میانگیر, مجموع_جابه_جایی, داده, (size / 4) * 4);
    }

    إذا (size % 4 != 0) {
        // If size is not a multiple of 4, we need to memset the remaining bytes
        size_t مانده_اندازه = size % 4;

        // pack the remaining bytes into a uint32_t
        uint32_t val32 = 0;

        لكل (size_t i = 0; i < مانده_اندازه; i++) {
            ((uint8_t *) &val32)[i] = ((ثابت uint8_t *) داده)[size - مانده_اندازه + i];
        }
        // memset the remaining bytes
        ggml_backend_webgpu_میانگیر_memset(buf_ctx->global_ctx, buf_ctx->میانگیر, val32,
                                          مجموع_جابه_جایی + (size - مانده_اندازه), مانده_اندازه);
    }
    WEBGPU_CPU_PROFILE_مجموع_انتها(مجموعه_tensor, buf_ctx->global_ctx);
}

ساكن فراغ ggml_backend_webgpu_میانگیر_get_tensor(ggml_backend_میانگیر_t میانگیر,
                                                  ثابت ggml_tensor *   tensor,
                                                  فراغ *                داده,
                                                  size_t                جابه_جایی,
                                                  size_t                size) {
    WEBGPU_CPU_PROFILE_مجموع_شروع(get_tensor);
    ggml_backend_webgpu_میانگیر_context * buf_ctx = (ggml_backend_webgpu_میانگیر_context *) میانگیر->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buf_ctx->label << ", " << tensor << ", " << داده
                                                              << ", " << جابه_جایی << ", " << size << ")");
    wgpu::Device device = buf_ctx->global_ctx->device;

    size_t مجموع_جابه_جایی = ggml_webgpu_tensor_جابه_جایی(tensor) + جابه_جایی;

    size_t local_جابه_جایی = مجموع_جابه_جایی % 4;
    إذا (local_جابه_جایی != 0) {
        // If offset is not a multiple of 4, we need to round it down to the previous
        // multiple of 4
        مجموع_جابه_جایی -= local_جابه_جایی;
    }

    size_t final_اندازه = size + local_جابه_جایی;
    إذا (final_اندازه % 4 != 0) {
        // If size is not a multiple of 4, we need to round it up to the next
        // multiple of 4
        final_اندازه += 4 - (final_اندازه % 4);
    }

    std::lock_guard<std::recursive_mutex> قفل(buf_ctx->global_ctx->mutex);

    إذا (buf_ctx->global_ctx->get_tensor_staging_buf == اشاره_تهی ||
        buf_ctx->global_ctx->get_tensor_staging_buf.Get_اندازه() < final_اندازه) {
        // Create a new staging buffer if it doesn't exist or is too small
        إذا (buf_ctx->global_ctx->get_tensor_staging_buf) {
            buf_ctx->global_ctx->get_tensor_staging_buf.Destroy();
        }
        ggml_webgpu_create_میانگیر(device, buf_ctx->global_ctx->get_tensor_staging_buf, final_اندازه,
                                  wgpu::میانگیر_Usage::CopyDst | wgpu::میانگیر_Usage::MapRead, "get_tensor_staging_buf");
    }

    // Copy the data from the buffer to the staging buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.Copy_میانگیر_مقصد_میانگیر(buf_ctx->میانگیر, مجموع_جابه_جایی, buf_ctx->global_ctx->get_tensor_staging_buf, 0,
                               final_اندازه);
    wgpu::Command_میانگیر commands = encoder.پایان();

    // Submit the command buffer to the queue
    buf_ctx->global_ctx->queue.Submit(1, &commands);

    // Map the staging buffer to read the data
    ggml_backend_webgpu_map_میانگیر(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,
                                   wgpu::Map_نما::Read, 0, final_اندازه);
    // Must specify size here since the staging buffer might be larger than the tensor size
    ثابت فراغ * mapped_range = buf_ctx->global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, final_اندازه);

    // Copy the data from the mapped range to the output buffer
    std::کپی_حافظه(داده, (ثابت فراغ *) ((ثابت حرف *) mapped_range + local_جابه_جایی), size);
    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();
    WEBGPU_CPU_PROFILE_مجموع_انتها(get_tensor, buf_ctx->global_ctx);
}

ساكن فراغ ggml_backend_webgpu_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << میانگیر << ", " << (uint32_t) value << ")");
    WEBGPU_CPU_PROFILE_مجموع_شروع(clear);
    ggml_backend_webgpu_میانگیر_context * buf_ctx = (ggml_backend_webgpu_میانگیر_context *) میانگیر->context;
    ggml_backend_webgpu_میانگیر_memset(buf_ctx->global_ctx, buf_ctx->میانگیر, value, 0, میانگیر->size);
    WEBGPU_CPU_PROFILE_مجموع_انتها(clear, buf_ctx->global_ctx);
}

ساكن ggml_backend_میانگیر_i ggml_backend_webgpu_میانگیر_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_میانگیر_free_میانگیر,
    /* .get_base        = */ ggml_backend_webgpu_میانگیر_get_پایه,
    /* .init_tensor     = */ عدم,  // TODO: optional, needed?
    /* .memset_tensor   = */ ggml_backend_webgpu_میانگیر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_میانگیر_مجموعه_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_میانگیر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ عدم,  // TODO: optional, implement this
    /* .clear           = */ ggml_backend_webgpu_میانگیر_clear,
    /* .reset           = */ عدم,  // TODO: optional, think it coordinates with
                                    // .init_tensor
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

ساكن ثابت حرف * ggml_backend_webgpu_میانگیر_type_get_name(ggml_backend_میانگیر_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    أرجع ctx->device_name.c_str();
}

ساكن ggml_backend_میانگیر_t ggml_backend_webgpu_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft,
                                                                          size_t                     size) {
    ساكن std::atomic<صحيح> میانگیر_شمار;
    صحيح                     میانگیر_id = میانگیر_شمار++;
    std::string             buf_name  = "tensor_buf" + std::to_string(میانگیر_id);
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer_" << میانگیر_id << ": " << size << " bytes");

    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::میانگیر_2                         buf;
    ggml_webgpu_create_میانگیر(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::میانگیر_Usage::Storage | wgpu::میانگیر_Usage::CopySrc | wgpu::میانگیر_Usage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_میانگیر_context * buf_ctx =
        جديد ggml_backend_webgpu_میانگیر_context(buf, buf_name, ctx->webgpu_global_ctx);

    أرجع ggml_backend_میانگیر_init(buft, ggml_backend_webgpu_میانگیر_interface, buf_ctx, size);
}

ساكن size_t ggml_backend_webgpu_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    أرجع dev_ctx->webgpu_global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment;
}

// maxBufferSize might be larger, but you can't bind more than
// maxStorageBufferBindingSize to a single binding.
ساكن size_t ggml_backend_webgpu_میانگیر_type_get_max_اندازه(ggml_backend_میانگیر_type_t buft) {
    ggml_backend_webgpu_device_context * dev_ctx =
        static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    أرجع dev_ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه;
}

ساكن size_t ggml_backend_webgpu_میانگیر_type_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft,
                                                             ثابت ggml_tensor *        tensor) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    size_t                               res = ggml_nbytes(tensor);
    بدل (tensor->op) {
        حالة GGML_OP_ARGSORT:
            res = ROUNDUP_POW2(res * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment,
                               WEBGPU_STORAGE_BUF_BINDING_MULT);
            اكسر;
        حالة GGML_OP_بالاترین_K:
            {
                ثابت ggml_tensor * src0 = tensor->src[0];
                إذا (src0) {
                    ثابت size_t پر = حجم(int32_t) * ggml_nelements(src0);
                    res               = ROUNDUP_POW2(
                        پر * 2 + ctx->webgpu_global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment,
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
                ثابت خودکار &        capabilities = ctx->webgpu_global_ctx->capabilities;
                إذا (ggml_webgpu_flash_attn_use_vec_مسیر(ctx->webgpu_global_ctx, Q, K, V)) {
                    ثابت منطقي kv_مستقیم =
                        ggml_webgpu_flash_attn_k_مستقیم(Q, K, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض) ||
                        ggml_webgpu_flash_attn_v_مستقیم(Q, V, GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض);
                    ثابت uint32_t kv_tile = ggml_webgpu_flash_attn_get_vec_kv_tile(
                        capabilities.limits.max_Compute_Workgroup_Storage_اندازه, (uint32_t) Q->ne[0], (uint32_t) V->ne[0],
                        ماسک != اشاره_تهی, kv_مستقیم);

                    ثابت uint32_t vec_nwg_cap = capabilities.min_subgroup_اندازه;
                    uint32_t       nwg = ggml_webgpu_flash_attn_vec_nwg(vec_nwg_cap, kv_tile, (uint32_t) K->ne[1]);

                    ثابت size_t   align = capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment;
                    ثابت uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
                    إذا (nwg > 1u) {
                        ثابت uint64_t tmp_داده_elems  = nrows * (uint64_t) V->ne[0] * nwg;
                        ثابت uint64_t tmp_stats_elems = nrows * 2u * nwg;
                        ثابت size_t   tmp_اندازه_بایتها = ROUNDUP_POW2((tmp_داده_elems + tmp_stats_elems) * حجم(عائم),
                                                                     WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += tmp_اندازه_بایتها + align;
                    } وإلا {
                        res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                    }
                    إذا (ماسک != اشاره_تهی) {
                        ثابت uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
                        ثابت uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
                        ثابت uint32_t stride_mask3    = (uint32_t) (ماسک->nb[3] / ggml_type_اندازه(ماسک->type));
                        ثابت uint32_t blk_batch_شمار = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
                        ثابت uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_شمار;
                        ثابت size_t   blk_اندازه_بایتها =
                            ROUNDUP_POW2(blk_elems * حجم(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
                        res += blk_اندازه_بایتها + align;
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
                    ggml_webgpu_can_use_mmvq(src0, src1, ctx->webgpu_global_ctx->capabilities.supports_dot_حاصلضرب,
                                             ctx->webgpu_global_ctx->vendor);
                إذا (use_mmvq) {
                    ثابت size_t q8_src1_اندازه = src1->ne[3] * src1->ne[2] * src1->ne[1] *
                                                (36 /* sizeof(q8_1) */ * (src1->ne[0] / /* block_size */ 32));
                    res = ROUNDUP_POW2(res + q8_src1_اندازه +
                                           ctx->webgpu_global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment,
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
                            ctx->webgpu_global_ctx->capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment * 2,
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

ساكن فراغ ggml_backend_webgpu_device_get_حافظه(ggml_backend_dev_t dev, size_t * حرر, size_t * مجموع) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: for now, return maxBufferSize as both free and total memory
    // Track https://github.com/gpuweb/gpuweb/issues/5505 for updates.
    uint64_t                             max_میانگیر_اندازه = ctx->webgpu_global_ctx->capabilities.limits.max_میانگیر_اندازه_2;
    // If we're on a 32-bit system, clamp to UINTPTR_MAX
#إذا UINTPTR_MAX < UINT64_MAX
    uint64_t max_ptr_اندازه = static_cast<uint64_t>(UINTPTR_MAX);
    إذا (max_میانگیر_اندازه > max_ptr_اندازه) {
        max_میانگیر_اندازه = max_ptr_اندازه;
    }
#پایان_اگر
    *حرر  = static_cast<size_t>(max_میانگیر_اندازه);
    *مجموع = static_cast<size_t>(max_میانگیر_اندازه);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_استفادهنشده(dev);
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_مجموع);
    props->caps = {
        /* .async                 = */ خطأ,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ خطأ,
        /* .mmap_support          = */ درست,
    };
}

ساكن ggml_guid_t ggml_backend_webgpu_guid(فراغ) {
    ساكن ggml_guid guid = { 0x67, 0xc7, 0xa4, 0xb1, 0x78, 0x74, 0x4f, 0x51,
                              0x9d, 0x65, 0x44, 0x6d, 0xe4, 0x1b, 0x82, 0x9a };
    أرجع &guid;
}

ساكن فراغ ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
    // we use the maximum workgroup size for the memset pipeline
    size_t max_نخها = ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup *
                         ctx->capabilities.limits.max_Compute_Workgroups_در_هر_Dimension;
    // Size the bytes_per_thread so that the largest buffer size can be handled
    ctx->capabilities.memset_بایتها_در_هر_thread =
        CEIL_DIV(ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه, max_نخها);
    std::vector<wgpu::Constant_مدخل> constants(2);
    constants[0].کلید     = "wg_size";
    constants[0].value   = ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
    constants[1].کلید     = "bytes_per_thread";
    constants[1].value   = ctx->capabilities.memset_بایتها_در_هر_thread;
    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
}

ساكن فراغ ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
    wgpu::RequestAdapterOptions options = {};

#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
    ثابت حرف * ثابت          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
    wgpu::DawnTogglesDescriptor adapterTogglesDesc;
    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;
    adapterTogglesDesc.enabled_Toggle_شمار = 2;
    options.بعدی_درون_Chain                   = &adapterTogglesDesc;
#پایان_اگر

    instance.WaitAny(
        instance.RequestAdapter(
            &options, wgpu::Callback_نما::AllowSpontaneous,
            [&adapter](wgpu::Request_Adapter_وضعیت وضعیت, wgpu::Adapter _adapter, wgpu::StringView پیام) {
                إذا (وضعیت != wgpu::Request_Adapter_وضعیت::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", std::string(پیام).c_str());
                    أرجع;
                }
                adapter = std::move(_adapter);
            }),
        UINT64_MAX);
}

ساكن فراغ create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
    ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, ctx->webgpu_global_ctx->adapter);
    GGML_ASSERT(ctx->webgpu_global_ctx->adapter != اشاره_تهی);

    ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);

    wgpu::AdapterInfo info{};
#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    wgpu::Adapter_Properties_Subgroup_ماتریس_Configs subgroup_ماتریس_configs{};
    إذا (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ماتریس)) {
        info.بعدی_درون_Chain = &subgroup_ماتریس_configs;
    }
#پایان_اگر
    ctx->webgpu_global_ctx->adapter.GetInfo(&info);
    ctx->webgpu_global_ctx->command_submit_batch_اندازه = ggml_backend_webgpu_get_command_submit_batch_اندازه();
    ctx->webgpu_global_ctx->max_inflight_batches      = ggml_backend_webgpu_get_max_inflight_batches();
    ctx->webgpu_global_ctx->vendor                    = info.vendor;
    ctx->webgpu_global_ctx->capabilities.supports_subgroups =
        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
    // for dot4I8packed
    ctx->webgpu_global_ctx->capabilities.supports_dot_حاصلضرب = ctx->webgpu_global_ctx->instance.HasWGSLLanguageFeature(
        wgpu::WGSLLanguageFeatureName::Packed4x8_Integer_Dot_حاصلضرب);

    منطقي valid_subgroup_ماتریس_config = خطأ;
#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    // Accept f16 subgroup matrix configurations (square or non-square).
    // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
    // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
    // The shaders are already parameterized to handle any M/N/K dimensions.
    إذا (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Chromium_Experimental_Subgroup_ماتریس)) {
        لكل (size_t i = 0; i < subgroup_ماتریس_configs.config_شمار; i++) {
            ثابت wgpu::Subgroup_ماتریس_Config config = subgroup_ماتریس_configs.configs[i];
            إذا (config.componentType == wgpu::Subgroup_ماتریس_Component_Type::F16 &&
                config.نتیجه_Component_Type == wgpu::Subgroup_ماتریس_Component_Type::F16) {
                ctx->webgpu_global_ctx->capabilities.sg_mat_m = config.M;
                ctx->webgpu_global_ctx->capabilities.sg_mat_n = config.N;
                ctx->webgpu_global_ctx->capabilities.sg_mat_k = config.K;
                valid_subgroup_ماتریس_config                  = درست;
                اكسر;
            }
        }
    }
#پایان_اگر
    ctx->webgpu_global_ctx->capabilities.supports_subgroup_ماتریس = valid_subgroup_ماتریس_config;

    // Runtime subgroup size can be any supported size in this range. Shaders
    // that allocate per-lane register arrays must size them for the minimum.
    ctx->webgpu_global_ctx->capabilities.min_subgroup_اندازه = info.subgroup_Min_اندازه;
    ctx->webgpu_global_ctx->capabilities.max_subgroup_اندازه = info.subgroup_Max_اندازه;
    // Initialize device
    std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };

#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
    إذا (ctx->webgpu_global_ctx->capabilities.supports_subgroup_ماتریس) {
        required_features.push_back(wgpu::FeatureName::Chromium_Experimental_Subgroup_ماتریس);
    }
#پایان_اگر

    إذا (ctx->webgpu_global_ctx->capabilities.supports_subgroups) {
        required_features.push_back(wgpu::FeatureName::Subgroups);
    }

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    required_features.push_back(wgpu::FeatureName::TimestampQuery);
#پایان_اگر

    wgpu::DeviceDescriptor dev_desc;
    dev_desc.requiredLimits       = &ctx->webgpu_global_ctx->capabilities.limits;
    dev_desc.requiredFeatures     = required_features.داده();
    dev_desc.required_Feature_شمار = required_features.size();
    dev_desc.مجموعه_Device_گمشده_Callback(
        wgpu::Callback_نما::AllowSpontaneous,
        [](ثابت wgpu::Device & device, wgpu::Device_گمشده_Reason reason, wgpu::StringView پیام) {
            إذا (reason == wgpu::Device_گمشده_Reason::Destroyed) {
                أرجع;
            }
            GGML_استفادهنشده(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<صحيح>(reason),
                           std::string(پیام).c_str());
        });
    dev_desc.مجموعه_Uncaptured_Error_Callback(
        [](ثابت wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView پیام) {
            GGML_استفادهنشده(device);
            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<صحيح>(reason),
                       std::string(پیام).c_str());
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
    deviceTogglesDesc.enabled_Toggle_شمار  = 3;
    deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
    deviceTogglesDesc.disabled_Toggle_شمار = 1;

    dev_desc.بعدی_درون_Chain = &deviceTogglesDesc;
#پایان_اگر

    ctx->webgpu_global_ctx->instance.WaitAny(
        ctx->webgpu_global_ctx->adapter.RequestDevice(
            &dev_desc, wgpu::Callback_نما::AllowSpontaneous,
            [ctx](wgpu::Request_Device_وضعیت وضعیت, wgpu::Device device, wgpu::StringView پیام) {
                إذا (وضعیت != wgpu::Request_Device_وضعیت::Success) {
                    GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", std::string(پیام).c_str());
                    أرجع;
                }
                ctx->webgpu_global_ctx->device = std::move(device);
            }),
        UINT64_MAX);
    GGML_ASSERT(ctx->webgpu_global_ctx->device != اشاره_تهی);

    ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
    ggml_webgpu_create_میانگیر(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                              WEBGPU_PARAMS_BUF_اندازه_بایتها, wgpu::میانگیر_Usage::CopyDst | wgpu::میانگیر_Usage::Uniform,
                              "memset_params_buf");
    ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.Get_صف();

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
        webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_اندازه_بایتها,
        webgpu_ctx->global_ctx->command_submit_batch_اندازه + WEBGPU_رقم_PARAM_جایگاه_SAFETY_MARGIN,
        webgpu_ctx->global_ctx->capabilities.limits.min_یکنواخت_میانگیر_جابه_جایی_Alignment);
    ggml_webgpu_create_میانگیر(webgpu_ctx->global_ctx->device, webgpu_ctx->مجموعه_ردیفها_dev_error_buf,
                              WEBGPU_مجموعه_ردیفها_ERROR_BUF_اندازه_بایتها,
                              wgpu::میانگیر_Usage::Storage | wgpu::میانگیر_Usage::CopySrc, "set_rows_dev_error_buf");
    ggml_webgpu_create_میانگیر(webgpu_ctx->global_ctx->device, webgpu_ctx->مجموعه_ردیفها_host_error_buf,
                              WEBGPU_مجموعه_ردیفها_ERROR_BUF_اندازه_بایتها,
                              wgpu::میانگیر_Usage::CopyDst | wgpu::میانگیر_Usage::MapRead, "set_rows_host_error_buf");

#اگر_تعریف_شده GGML_WEBGPU_GPU_PROFILE
    webgpu_ctx->batch_compute_passes = خطأ;
    ggml_webgpu_create_میانگیر(
        webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_dev_buf, WEBGPU_TIMESTAMP_QUERY_BUF_اندازه_بایتها,
        wgpu::میانگیر_Usage::QueryResolve | wgpu::میانگیر_Usage::CopySrc, "profile_timestamp_dev_buf");
    ggml_webgpu_create_میانگیر(webgpu_ctx->global_ctx->device, webgpu_ctx->profile_timestamp_host_buf,
                              WEBGPU_TIMESTAMP_QUERY_BUF_اندازه_بایتها,
                              wgpu::میانگیر_Usage::CopyDst | wgpu::میانگیر_Usage::MapRead, "profile_timestamp_host_buf");
    wgpu::Query_مجموعه_Descriptor query_مجموعه_desc = {};
    query_مجموعه_desc.type                     = wgpu::QueryType::Timestamp;
    query_مجموعه_desc.count                    = WEBGPU_MAX_PROFILE_QUERY_شمار;
    webgpu_ctx->profile_timestamp_query_مجموعه = webgpu_ctx->global_ctx->device.Create_Query_مجموعه(&query_مجموعه_desc);
#پایان_اگر

#اگر_تعریف_شده GGML_WEBGPU_DEBUG
    // Initialize debug buffers
    ggml_webgpu_create_میانگیر(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_host_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * حجم(uint32_t),
                              wgpu::میانگیر_Usage::CopyDst | wgpu::میانگیر_Usage::MapRead, "debug_host_buf");
    ggml_webgpu_create_میانگیر(webgpu_ctx->global_ctx->device, webgpu_ctx->global_ctx->debug_dev_buf,
                              WEBGPU_DEBUG_BUF_ELEMS * حجم(uint32_t),
                              wgpu::میانگیر_Usage::Storage | wgpu::میانگیر_Usage::CopySrc, "debug_dev_buf");
#پایان_اگر
    أرجع webgpu_ctx;
}

ساكن ggml_backend_t ggml_backend_webgpu_backend_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_استفادهنشده(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_backend_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    خودکار * backend_ctx      = جديد ggml_backend_webgpu_context();
    backend_ctx->name       = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx->webgpu_ctx = initialize_webgpu_context(dev);

    // See GGML Backend Interface section
    خودکار * backend = جديد ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };
    أرجع backend;
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_webgpu_device_get_میانگیر_type(ggml_backend_dev_t dev) {
    // See GGML Backend Buffer Type Interface section

    ساكن بنية ggml_backend_میانگیر_type ggml_backend_webgpu_میانگیر_type = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_webgpu_میانگیر_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_webgpu_میانگیر_type_alloc_میانگیر,
                        /* .alloc_buffer_n      = */ عدم,
                        /* .get_alignment       = */ ggml_backend_webgpu_میانگیر_type_get_alignment,
                        /* .get_max_size        = */ ggml_backend_webgpu_میانگیر_type_get_max_اندازه,
                        /* .get_alloc_size      = */ ggml_backend_webgpu_میانگیر_type_get_alloc_اندازه,
                        /* .get_alloc_size_n    = */ عدم,
                        /* .is_host             = */ عدم,  // defaults to false
        },
        /* .device  = */
        dev,
        /* .context = */ عدم
    };

    أرجع &ggml_backend_webgpu_میانگیر_type;
}

ساكن منطقي ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_میانگیر_type_t buft) {
    GGML_استفادهنشده(dev);
    أرجع buft->iface.get_name == ggml_backend_webgpu_میانگیر_type_get_name;
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
            أرجع درست;
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
    إذا (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه ||
        (src0 != اشاره_تهی &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه) ||
        (src1 != اشاره_تهی &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه)) {
        أرجع خطأ;
    }

    منطقي supports_op = خطأ;
    بدل (op->op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_جایگشت:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE:
            supports_op = درست;
            اكسر;
        حالة GGML_OP_جمع_کن:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (src0->type == op->type) &&
                          (src1->type == op->type);
            اكسر;
        حالة GGML_OP_جمع_کن_ID:
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
        حالة GGML_OP_مجموعه:
            supports_op = src0->type == src1->type && src0->type == op->type &&
                          (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32);
            اكسر;
        حالة GGML_OP_مجموعه_ردیفها:
            supports_op = ((op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_Q8_0 ||
                            op->type == GGML_TYPE_Q4_0) &&
                           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                           (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32));
            اكسر;
        حالة GGML_OP_GET_ردیفها:
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
                                supports_op = درست;
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
                            supports_op = درست;
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
                ثابت خودکار & capabilities             = ctx->webgpu_global_ctx->capabilities;
                ثابت size_t storage_جابه_جایی_alignment = capabilities.limits.min_Storage_میانگیر_جابه_جایی_Alignment;

                // subgroup matrix path requirements
                ثابت منطقي use_subgroup_ماتریس = ggml_webgpu_flash_attn_can_use_subgroup_ماتریس_مسیر(
                    capabilities.supports_subgroup_ماتریس, capabilities.sg_mat_k, capabilities.sg_mat_n, src0, src2);

                // tile path requirements
                ثابت منطقي float_vec4_aligned =
                    ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src1, storage_جابه_جایی_alignment)) &&
                    ((src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_F32) ||
                     ggml_webgpu_flash_attn_float_vec4_aligned(src2, storage_جابه_جایی_alignment));
                ثابت uint32_t k_tile_سر_align = (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض :
                                                       (uint32_t) ggml_blck_اندازه(src1->type);
                ثابت uint32_t v_tile_سر_align = (src2->type == GGML_TYPE_F32 || src2->type == GGML_TYPE_F16) ?
                                                       GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_عرض :
                                                       (uint32_t) ggml_blck_اندازه(src2->type);
                ثابت منطقي     tile_kv_سر_dims_aligned =
                    src0->ne[0] % k_tile_سر_align == 0 && src2->ne[0] % v_tile_سر_align == 0;
                ثابت منطقي tile_can_dispatch_all_q_ردیفها =
                    capabilities.limits.max_Compute_Invocations_در_هر_Workgroup >=
                    GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE * capabilities.max_subgroup_اندازه;
                ثابت منطقي use_tile = !use_subgroup_ماتریس && capabilities.supports_subgroups && float_vec4_aligned &&
                                      tile_kv_سر_dims_aligned && tile_can_dispatch_all_q_ردیفها;

                إذا (!use_subgroup_ماتریس && !use_tile) {
                    supports_op = خطأ;
                    اكسر;
                }
                ثابت uint32_t q_tile =
                    use_subgroup_ماتریس ? capabilities.sg_mat_m : GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
                ثابت uint32_t kv_granularity = use_subgroup_ماتریس ? capabilities.sg_mat_n : 1u;
                ثابت منطقي     kv_مستقیم = use_subgroup_ماتریس ?
                                               ggml_webgpu_flash_attn_k_مستقیم(src0, src1, capabilities.sg_mat_k) ||
                                                   ggml_webgpu_flash_attn_v_مستقیم(src0, src2, capabilities.sg_mat_k) :
                                               خطأ;
                ثابت uint32_t max_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(
                    capabilities.limits.max_Compute_Workgroup_Storage_اندازه, q_tile, kv_granularity, (uint32_t) src0->ne[0],
                    (uint32_t) src2->ne[0], op->src[3] != اشاره_تهی, kv_مستقیم);
                supports_op = max_kv_tile > 0;
                اكسر;
            }
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
            supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_ردیفها(src0);
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
                حالة GGML_GLU_OP_GEGLU_سریع:
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
                    حالة GGML_UNARY_OP_گام:
                    حالة GGML_UNARY_OP_TANH:
                    حالة GGML_UNARY_OP_ELU:
                    حالة GGML_UNARY_OP_RELU:
                    حالة GGML_UNARY_OP_SIGMOID:
                    حالة GGML_UNARY_OP_GELU:
                    حالة GGML_UNARY_OP_GELU_سریع:
                    حالة GGML_UNARY_OP_SILU:
                    حالة GGML_UNARY_OP_HARDSWISH:
                    حالة GGML_UNARY_OP_HARDSIGMOID:
                    حالة GGML_UNARY_OP_EXP:
                    حالة GGML_UNARY_OP_GELU_ERF:
                    حالة GGML_UNARY_OP_SOFTPLUS:
                    حالة GGML_UNARY_OP_EXPM1:
                    حالة GGML_UNARY_OP_FLOOR:
                    حالة GGML_UNARY_OP_CEIL:
                    حالة GGML_UNARY_OP_دور:
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
        حالة GGML_OP_SSM_پیمایش:
            supports_op = op->type == GGML_TYPE_F32 &&
                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
            اكسر;
        حالة GGML_OP_GATED_DELTA_NET:
            {
                ثابت uint32_t s_v = (uint32_t) src2->ne[0];
                supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                              src2->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                              op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
                              s_v <= ctx->webgpu_global_ctx->capabilities.limits.max_Compute_Invocations_در_هر_Workgroup;
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
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_ردیفها(src0);
            اكسر;
        حالة GGML_OP_بالاترین_K:
            supports_op = op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_ردیفها(src0);
            اكسر;
        حالة GGML_OP_CUMSUM:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type;
            اكسر;
        حالة GGML_OP_جمع:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
            اكسر;
        حالة GGML_OP_جمع_ردیفها:
            supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_ردیفها(src0);
            اكسر;
        حالة GGML_OP_UPSCALE:
            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
            اكسر;
        افتراضي:
            اكسر;
    }
    إذا (ggml_nbytes(op) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه ||
        (src0 != اشاره_تهی &&
         ggml_nbytes(src0) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه) ||
        (src1 != اشاره_تهی &&
         ggml_nbytes(src1) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه) ||
        (src2 != اشاره_تهی &&
         ggml_nbytes(src2) > ctx->webgpu_global_ctx->capabilities.limits.max_Storage_میانگیر_Binding_اندازه)) {
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
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_میانگیر_type,
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

ساكن size_t ggml_backend_webgpu_reg_get_device_شمار(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    أرجع ctx->device_شمار;
}

// Only one device is supported for now
ساكن ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t شاخص) {
    GGML_ASSERT(شاخص == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    WEBGPU_CPU_PROFILE_مجموع_شروع(reg_get_device);

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

    WEBGPU_CPU_PROFILE_مجموع_انتها(reg_get_device, reg_ctx->webgpu_global_ctx);
    أرجع &device;
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_شمار,
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
    ctx->device_شمار = 0;

    // Keep one Dawn/WebGPU instance alive for the lifetime of the static backend
    // registry. Recreating it on repeated registry lookups can invalidate
    // adapter/device references that are still held by the backend/device layer.
    إذا (ctx->webgpu_global_ctx != اشاره_تهی && ctx->webgpu_global_ctx->instance != اشاره_تهی) {
        أرجع &reg;
    }

    wgpu::InstanceDescriptor               instance_descriptor{};
    std::vector<wgpu::InstanceFeatureName> instance_features = { wgpu::InstanceFeatureName::TimedWaitAny };
    instance_descriptor.requiredFeatures                     = instance_features.داده();
    instance_descriptor.required_Feature_شمار                 = instance_features.size();

#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
    ثابت حرف * ثابت          instanceEnabledToggles[] = { "allow_unsafe_apis" };
    wgpu::DawnTogglesDescriptor instanceTogglesDesc;
    instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
    instanceTogglesDesc.enabled_Toggle_شمار = 1;
    instance_descriptor.بعدی_درون_Chain        = &instanceTogglesDesc;
#پایان_اگر

    wgpu::Instance inst              = wgpu::CreateInstance(&instance_descriptor);
    ctx->webgpu_global_ctx           = webgpu_global_context(جديد webgpu_global_context_struct());
    ctx->webgpu_global_ctx->instance = std::move(inst);

    // Probe for adapter support
    wgpu::Adapter adapter;
    إذا (ctx->webgpu_global_ctx->instance != اشاره_تهی) {
        ggml_backend_webgpu_request_adapter(ctx->webgpu_global_ctx->instance, adapter);
    }

    // WebGPU backend requires f16 support and, on native, implicit device synchronization.
    إذا (adapter != اشاره_تهی && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
#إذا !defined(__EMSCRIPTEN__) && !defined(__wasi__)
        && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
#پایان_اگر
    ) {
        ctx->device_شمار = 1;
    }

    أرجع &reg;
}

ggml_backend_t ggml_backend_webgpu_init(فراغ) {
    ggml_backend_reg_t reg = ggml_backend_webgpu_reg();
    إذا (ggml_backend_reg_dev_شمار(reg) == 0) {
        أرجع اشاره_تهی;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    أرجع ggml_backend_webgpu_backend_init(dev, اشاره_تهی);
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)
