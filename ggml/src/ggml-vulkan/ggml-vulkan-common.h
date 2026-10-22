#ہدایت once
#شامل "ggml-vulkan-push-constants.h"

اندرونی std::ostream & عامل<<(std::ostream & os, vk::بفر_2 بفر) {
    واپس os << static_cast<Vk_بفر>(بفر);
}

// shared globals
خارجی ggml_backend_بفر_type_i ggml_backend_vk_بفر_type_interface;
خارجی bool vk_حافظہ_logger_enabled;
خارجی bool vk_perf_logger_enabled;
خارجی bool vk_perf_logger_concurrent;
خارجی bool vk_enable_sync_logger;
خارجی uint32_t vk_perf_logger_frequency;
خارجی std::string vk_pipeline_stats_filter;
خارجی خالی * ثابت vk_ptr_بنیاد;
خارجی vk_instance_t vk_instance;
خارجی ggml_backend_بفر_i ggml_backend_vk_بفر_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
خالی ggml_vk_instance_init();
خالی ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
صحیح ggml_vk_get_device_گنتی();
خالی ggml_vk_get_device_description(صحیح device, حرف * description, size_t description_سائز);
bool ggml_vk_instance_layer_settings_دستیاب();
bool ggml_vk_instance_portability_enumeration_ext_دستیاب(ثابت std::سمتیہ<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_دستیاب(ثابت std::سمتیہ<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ثابت vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_میٹرکس_support(ثابت vk::PhysicalDeviceProperties& props, ثابت vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_گنتی(ثابت vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_اندر_range(uint32_t driver_نسخہ, uint32_t زیریں_major, uint32_t زیریں_minor, uint32_t بالائی_major, uint32_t بالائی_minor);

// shaders
خالی ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_صفیں, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_نقاب, bool use_نقاب_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_سائز(ثابت std::string &pipeline_name, ثابت vk_device_architecture &arch);
خالی ggml_vk_بھرو_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_بفر ggml_vk_create_بفر_check(vk_device& device, size_t سائز, vk::حافظہ_Property_Flags req_flags, vk::حافظہ_Property_Flags fallback_flags = vk::حافظہ_Property_Flags(0));
vk_بفر ggml_vk_create_بفر_device(vk_device& device, size_t سائز);
خالی ggml_vk_destroy_بفر(vk_بفر& buf);
خالی * ggml_vk_host_malloc(size_t سائز);
خالی ggml_vk_host_free(خالی * ptr);
خالی ggml_vk_host_get(ثابت vk_device& device, ثابت خالی * ptr, vk_بفر& buf, size_t& buf_انحراف);
خالی ggml_vk_ensure_sync_staging_بفر(vk_device& device, size_t سائز);
خالی ggml_vk_ensure_sync_staging_بفر(ggml_backend_vk_context * ctx, size_t سائز);
bool ggml_vk_بفر_write_2d_async(vk_context subctx, vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی, bool sync_staging = false);
bool ggml_vk_بفر_write_async(vk_context subctx, vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t سائز, bool sync_staging = false);
خالی ggml_vk_بفر_write_2d(vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی);
خالی ggml_vk_بفر_write(vk_بفر& dst, size_t انحراف, ثابت خالی * src, size_t سائز);
bool ggml_vk_بفر_read_2d_async(vk_context subctx, vk_بفر& src, size_t انحراف, خالی * dst, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی, bool sync_staging = false);
خالی ggml_vk_بفر_read_2d(vk_بفر& src, size_t انحراف, خالی * dst, size_t spitch, size_t dpitch, size_t چوڑائی, size_t اونچائی);
خالی ggml_vk_بفر_read(vk_بفر& src, size_t انحراف, خالی * dst, size_t سائز);
خالی ggml_vk_بفر_copy_async(vk_context& ctx, vk_بفر& dst, size_t dst_انحراف, vk_بفر& src, size_t src_انحراف, size_t سائز);
خالی ggml_vk_بفر_copy(vk_بفر& dst, size_t dst_انحراف, vk_بفر& src, size_t src_انحراف, size_t سائز);
خالی ggml_vk_بفر_memset_async(vk_context& ctx, vk_بفر& dst, size_t انحراف, uint32_t c, size_t سائز);
خالی ggml_vk_بفر_memset(vk_بفر& dst, size_t انحراف, uint32_t c, size_t سائز);
vk_بفر ggml_vk_بفر_from_host_ptr(vk_device & device, خالی * ptr, size_t سائز);

// pipelines
uint64_t vk_tensor_انحراف(ثابت ggml_tensor * tensor);
uint32_t get_misalign_بائٹیں(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
خالی ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
خالی ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
خالی ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
خالی ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_ڈھونڈو_قطار_family_اشاریہ(std::سمتیہ<vk::قطار_Family_Properties>& قطار_family_props, ثابت vk::قطار_Flags& required, ثابت vk::قطار_Flags& avoid, int32_t compute_اشاریہ, uint32_t min_عدد_queues);
std::unique_ptr<vk_قطار> ggml_vk_create_قطار(vk_device& device, uint32_t قطار_family_اشاریہ, uint32_t قطار_اشاریہ, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_قطار> ggml_vk_create_aliased_قطار(vk_device& device, ثابت std::unique_ptr<vk_قطار>& ماخذ);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
خالی ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
خالی ggml_vk_قطار_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ثابت ggml_backend_vk_context* ctx, ثابت vk_بفر& buf, size_t انحراف = 0);
خالی ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
خالی ggml_vk_سیٹ_event(vk_context& ctx, vk::Event& event);
خالی ggml_vk_wait_events(vk_context& ctx, std::سمتیہ<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * tensor, bool allow_misalign = false);
خالی ggml_vk_cmd_label_begin(vk::Command_بفر buf, ثابت حرف * name);
خالی ggml_vk_ctx_انتہا(vk_context& ctx);
خالی ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_سائز(size_t چوڑائی, size_t align);
خالی deferred_memcpy(خالی * dst, ثابت خالی * src, size_t سائز, std::سمتیہ<vk_staging_memcpy>* memcpys = nullptr);
خالی deferred_memset(خالی * dst, uint32_t val, size_t سائز, std::سمتیہ<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_مقصد_fp16(ggml_backend_vk_context * ctx, ggml_type type);
خالی ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& بانٹو_k_بفر, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t بانٹو_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_ستون_بائٹیں, uint64_t d_ستون_بائٹیں);
bool ggml_vk_dim01_contiguous(ثابت ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src, ثابت ggml_tensor * dst, ggml_type مقصد);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
خالی ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت vk_subbuffer & اندر, ثابت vk_subbuffer & برآمد, uint32_t ne);
خالی ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * mixes, ثابت ggml_tensor * scale, ثابت ggml_tensor * بنیاد, ggml_tensor * dst);
خالی ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * weights, ggml_tensor * dst);
خالی ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * residual, ثابت ggml_tensor * post, ثابت ggml_tensor * comb, ggml_tensor * dst, ثابت ggml_tensor * gate_scale_اندر = nullptr);
خالی ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
bool ggml_vk_use_mul_mat_vec_id(ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);

// flash-attn
خالی ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * q, ثابت ggml_tensor * k, ثابت ggml_tensor * v, ثابت ggml_tensor * نقاب, ثابت ggml_tensor * sinks, ggml_tensor * dst);

// operators
خالی ggml_vk_cpy_مقصد_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor, ثابت vk_subbuffer & اندر, ثابت vk_subbuffer & برآمد);
bool ggml_vk_can_use_fwht(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);
خالی ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src, ggml_tensor * dst);
خالی ggml_vk_get_صفیں(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_get_صفیں_عقب(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_multi_جوڑو(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_جوڑو(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_برآمد_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
صحیح ggml_vk_unary_mul_op_اشاریہ(ggml_unary_op op);
خالی ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_جوڑو_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
خالی ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_gated_خطی_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_ssm_جائزہ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_opt_قدم_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_opt_قدم_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
خالی ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_repeat_عقب(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_سیٹ_صفیں(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_silu_عقب(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_گروہ_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_سائز(ggml_backend_vk_context * ctx, ثابت ggml_tensor *گرہ);
خالی ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, اعشاری * op_params);
خالی ggml_vk_rms_norm_عقب(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_diag_نقاب_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
خالی ggml_vk_soft_max_عقب(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحیح گرہ_idx, bool backprop);
خالی ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_مجموع(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_مجموع_صفیں(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_اوسط(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_cross_entropy_loss_عقب(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
خالی ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_گنتی_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
خالی ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
خالی ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);

//گراف
خالی ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_گراف(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحیح گرہ_idx, ggml_tensor *گرہ_begin, صحیح گرہ_idx_begin, bool آخری_گرہ, bool almost_ready, bool submit);
خالی ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, صحیح tensor_idx, bool almost_ready);
خالی ggml_vk_گراف_cleanup(ggml_backend_vk_context * ctx);
خالی ggml_vk_cleanup(ggml_backend_vk_context * ctx);
خالی ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_خالی_خانے(ggml_tensor * گرہ);
bool ggml_vk_can_fuse(ثابت ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, std::initializer_list<شمار ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ثابت ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, صحیح عدد_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, topk_moe_منوال منوال);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
bool ggml_vk_can_fuse_rope_سیٹ_صفیں(ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
bool ggml_vk_can_fuse_rms_norm_سیٹ_صفیں(ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
bool ggml_vk_tensors_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
uint32_t ggml_vk_fuse_multi_جوڑو(ggml_backend_vk_context * ctx, ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx);
خالی ggml_vk_گراف_optimize(ggml_backend_t backend, ساخت ggml_cgraph * گراف, ساخت ggml_backend_گراف_optimize_params * params);

// backend
bool ggml_backend_بفر_is_vk(ggml_backend_بفر_t بفر);
خالی ggml_backend_vk_بفر_free_بفر(ggml_backend_بفر_t بفر);
خالی * ggml_backend_vk_بفر_get_بنیاد(ggml_backend_بفر_t بفر);
شمار ggml_حالت ggml_backend_vk_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor);
خالی ggml_backend_vk_بفر_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t قدر, size_t انحراف, size_t سائز);
خالی ggml_backend_vk_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز);
خالی ggml_backend_vk_بفر_سیٹ_tensor_2d(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا);
خالی ggml_backend_vk_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t سائز);
خالی ggml_backend_vk_بفر_get_tensor_2d(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t سائز, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا);
bool ggml_backend_vk_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst);
خالی ggml_backend_vk_بفر_clear(ggml_backend_بفر_t بفر, uint8_t قدر);
ثابت حرف * ggml_backend_vk_بفر_type_name(ggml_backend_بفر_type_t buft);
ggml_backend_بفر_t ggml_backend_vk_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t سائز);
size_t ggml_backend_vk_بفر_type_get_alignment(ggml_backend_بفر_type_t buft);
size_t ggml_backend_vk_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft);
size_t ggml_backend_vk_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor);
خالی ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_سائز(ثابت ggml_tensor * op);
#اگر_تعریف_شدہ GGML_VULKAN_تسلسل_TESTS
سانچہ <typename X_TYPE, typename Y_TYPE>
خالی ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t عدد_it, صحیح بانٹو_k, صحیح shader_سائز);
خالی ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
خالی ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t عدد_it, size_t بانٹو_k, size_t shader_سائز, ggml_type quant, bool mmq = false);
#اگر_ختم
#اگر_تعریف_شدہ GGML_VULKAN_CHECK_RESULTS
خارجی size_t vk_skip_checks;
خارجی size_t vk_output_tensor;
خالی ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحیح tensor_idx);
خالی ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحیح tensor_idx);
#اگر_ختم

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
خالی ggml_vk_print_device_fault_info(ثابت vk_device& device);
uint64_t ggml_vk_get_گرہ_flops(ثابت ggml_tensor * گرہ);
خالی ggml_vk_print_گرہ_list(ثابت ggml_cgraph * cgraph, صحیح آغاز, صحیح انتہا);
خالی ggml_vk_print_device_گمشدہ_info(ثابت vk_device& device);
size_t ggml_vk_tensor_بفر_انحراف(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
size_t ggml_vk_descriptor_انحراف(size_t tensor_انحراف, size_t alignment, size_t type_سائز);
uint32_t ggml_vk_concat_unit_سائز(ggml_type type);
bool ggml_vk_concat_supported(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);

سانچہ <typename T>
اندرونی خالی ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_بفر_Info> ثابت& descriptor_بفر_infos, ثابت T &ڈالو_constants, std::array<uint32_t, 3> elements) {
    ثابت uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ثابت uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ثابت uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    برائے (خودکار& بفر : descriptor_بفر_infos) {
        std::cerr << "(" << بفر.بفر << ", " << بفر.انحراف << ", " << بفر.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_گروہ_گنتی[2]);
    GGML_ASSERT(ctx->descriptor_سیٹ_idx < ctx->descriptor_sets.سائز());
    GGML_ASSERT(descriptor_بفر_infos.سائز() <= MAX_PARAMETER_گنتی);
    GGML_ASSERT(pipeline->parameter_گنتی == descriptor_بفر_infos.سائز());
    GGML_ASSERT(pipeline->ڈالو_constant_سائز == ڈالو_constant_سائز(ڈالو_constants));

    ثابت uint32_t descriptor_سیٹ_idx = ctx->descriptor_سیٹ_idx++;
    vk::Descriptor_سیٹ& descriptor_سیٹ = ctx->descriptor_sets[descriptor_سیٹ_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ثابت uint64_t destroy_گنتی = ctx->device->بفر_destroy_گنتی.بھرو(std::memory_order_acquire);
    اگر (ctx->descriptor_سیٹ_bindings_destroy_گنتی != destroy_گنتی) {
        برائے (خودکار & b : ctx->descriptor_سیٹ_bindings) {
            b.clear();
        }
        ctx->descriptor_سیٹ_bindings_destroy_گنتی = destroy_گنتی;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::سمتیہ<vk::Descriptor_بفر_Info> & bindings = ctx->descriptor_سیٹ_bindings[descriptor_سیٹ_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.سائز() == descriptor_بفر_infos.سائز();
    اگر (same) {
        size_t i = 0;
        برائے (ثابت vk::Descriptor_بفر_Info & info : descriptor_بفر_infos) {
            ثابت vk::Descriptor_بفر_Info & پچھلا = bindings[i++];
            اگر (پچھلا.بفر != info.بفر || پچھلا.انحراف != info.انحراف || پچھلا.range != info.range) {
                same = false;
                توڑو;
            }
        }
    }
    اگر (!same) {
        vk::Write_Descriptor_سیٹ write_descriptor_سیٹ{ descriptor_سیٹ, 0, 0, pipeline->parameter_گنتی, vk::DescriptorType::e_Storage_بفر, nullptr, descriptor_بفر_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_سیٹ }, {});
        bindings.تفویض(descriptor_بفر_infos.begin(), descriptor_بفر_infos.انتہا());
    }

    subctx->s->بفر->buf.ڈالو_Constants(pipeline->layout, vk::Shader_Stage_Flag_بٹ::eCompute, 0, ڈالو_constant_سائز(ڈالو_constants), ڈالو_constant_ڈیٹا(ڈالو_constants));
    subctx->s->بفر->buf.bindPipeline(vk::Pipeline_Bind_نقطہ::eCompute, pipeline->pipeline);
    subctx->s->بفر->buf.bindDescriptorSets(vk::Pipeline_Bind_نقطہ::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_سیٹ },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->بفر->buf.dispatch(wg0, wg1, wg2);
    }
}

