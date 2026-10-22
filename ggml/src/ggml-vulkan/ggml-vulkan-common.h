#رهنمود once
#ضمن "ggml-vulkan-push-constants.h"

مضمن std::ostream & عملگر<<(std::ostream & os, vk::میانگیر_2 میانگیر) {
    أرجع os << static_cast<Vk_میانگیر>(میانگیر);
}

// shared globals
خارجي ggml_backend_میانگیر_type_i ggml_backend_vk_میانگیر_type_interface;
خارجي bool vk_حافظه_logger_enabled;
خارجي bool vk_perf_logger_enabled;
خارجي bool vk_perf_logger_concurrent;
خارجي bool vk_enable_sync_logger;
خارجي uint32_t vk_perf_logger_frequency;
خارجي std::string vk_pipeline_stats_filter;
خارجي فراغ * ثابت vk_ptr_پایه;
خارجي vk_instance_t vk_instance;
خارجي ggml_backend_میانگیر_i ggml_backend_vk_میانگیر_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
فراغ ggml_vk_instance_init();
فراغ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
صحيح ggml_vk_get_device_شمار();
فراغ ggml_vk_get_device_description(صحيح device, حرف * description, size_t description_اندازه);
bool ggml_vk_instance_layer_settings_موجود();
bool ggml_vk_instance_portability_enumeration_ext_موجود(ثابت std::وکتور<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_موجود(ثابت std::وکتور<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ثابت vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_ماتریس_support(ثابت vk::PhysicalDeviceProperties& props, ثابت vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_شمار(ثابت vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_درون_range(uint32_t driver_نسخه, uint32_t پایینی_major, uint32_t پایینی_minor, uint32_t فوقانی_major, uint32_t فوقانی_minor);

// shaders
فراغ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ردیفها, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_ماسک, bool use_ماسک_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_اندازه(ثابت std::string &pipeline_name, ثابت vk_device_architecture &arch);
فراغ ggml_vk_بارگیری_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_میانگیر ggml_vk_create_میانگیر_check(vk_device& device, size_t اندازه, vk::حافظه_Property_Flags req_flags, vk::حافظه_Property_Flags fallback_flags = vk::حافظه_Property_Flags(0));
vk_میانگیر ggml_vk_create_میانگیر_device(vk_device& device, size_t اندازه);
فراغ ggml_vk_destroy_میانگیر(vk_میانگیر& buf);
فراغ * ggml_vk_host_malloc(size_t اندازه);
فراغ ggml_vk_host_free(فراغ * ptr);
فراغ ggml_vk_host_get(ثابت vk_device& device, ثابت فراغ * ptr, vk_میانگیر& buf, size_t& buf_جابه_جایی);
فراغ ggml_vk_ensure_sync_staging_میانگیر(vk_device& device, size_t اندازه);
فراغ ggml_vk_ensure_sync_staging_میانگیر(ggml_backend_vk_context * ctx, size_t اندازه);
bool ggml_vk_میانگیر_write_2d_async(vk_context subctx, vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع, bool sync_staging = false);
bool ggml_vk_میانگیر_write_async(vk_context subctx, vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t اندازه, bool sync_staging = false);
فراغ ggml_vk_میانگیر_write_2d(vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع);
فراغ ggml_vk_میانگیر_write(vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t اندازه);
bool ggml_vk_میانگیر_read_2d_async(vk_context subctx, vk_میانگیر& src, size_t جابه_جایی, فراغ * dst, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع, bool sync_staging = false);
فراغ ggml_vk_میانگیر_read_2d(vk_میانگیر& src, size_t جابه_جایی, فراغ * dst, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع);
فراغ ggml_vk_میانگیر_read(vk_میانگیر& src, size_t جابه_جایی, فراغ * dst, size_t اندازه);
فراغ ggml_vk_میانگیر_copy_async(vk_context& ctx, vk_میانگیر& dst, size_t dst_جابه_جایی, vk_میانگیر& src, size_t src_جابه_جایی, size_t اندازه);
فراغ ggml_vk_میانگیر_copy(vk_میانگیر& dst, size_t dst_جابه_جایی, vk_میانگیر& src, size_t src_جابه_جایی, size_t اندازه);
فراغ ggml_vk_میانگیر_memset_async(vk_context& ctx, vk_میانگیر& dst, size_t جابه_جایی, uint32_t c, size_t اندازه);
فراغ ggml_vk_میانگیر_memset(vk_میانگیر& dst, size_t جابه_جایی, uint32_t c, size_t اندازه);
vk_میانگیر ggml_vk_میانگیر_from_host_ptr(vk_device & device, فراغ * ptr, size_t اندازه);

// pipelines
uint64_t vk_tensor_جابه_جایی(ثابت ggml_tensor * tensor);
uint32_t get_misalign_بایتها(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
فراغ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
فراغ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
فراغ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
فراغ ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_پیدا_کن_صف_family_شاخص(std::وکتور<vk::صف_Family_Properties>& صف_family_props, ثابت vk::صف_Flags& required, ثابت vk::صف_Flags& avoid, int32_t compute_شاخص, uint32_t min_رقم_queues);
std::unique_ptr<vk_صف> ggml_vk_create_صف(vk_device& device, uint32_t صف_family_شاخص, uint32_t صف_شاخص, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_صف> ggml_vk_create_aliased_صف(vk_device& device, ثابت std::unique_ptr<vk_صف>& منبع);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
فراغ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
فراغ ggml_vk_صف_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ثابت ggml_backend_vk_context* ctx, ثابت vk_میانگیر& buf, size_t جابه_جایی = 0);
فراغ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
فراغ ggml_vk_مجموعه_event(vk_context& ctx, vk::Event& event);
فراغ ggml_vk_wait_events(vk_context& ctx, std::وکتور<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * tensor, bool allow_misalign = false);
فراغ ggml_vk_cmd_label_begin(vk::Command_میانگیر buf, ثابت حرف * name);
فراغ ggml_vk_ctx_انتها(vk_context& ctx);
فراغ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_اندازه(size_t عرض, size_t align);
فراغ deferred_memcpy(فراغ * dst, ثابت فراغ * src, size_t اندازه, std::وکتور<vk_staging_memcpy>* memcpys = nullptr);
فراغ deferred_memset(فراغ * dst, uint32_t val, size_t اندازه, std::وکتور<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_مقصد_fp16(ggml_backend_vk_context * ctx, ggml_type type);
فراغ ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& تقسیم_k_میانگیر, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t تقسیم_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_ستون_بایتها, uint64_t d_ستون_بایتها);
bool ggml_vk_dim01_contiguous(ثابت ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src, ثابت ggml_tensor * dst, ggml_type مقصد);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
فراغ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت vk_subbuffer & درون, ثابت vk_subbuffer & بیرون, uint32_t ne);
فراغ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * mixes, ثابت ggml_tensor * scale, ثابت ggml_tensor * پایه, ggml_tensor * dst);
فراغ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * weights, ggml_tensor * dst);
فراغ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * residual, ثابت ggml_tensor * post, ثابت ggml_tensor * comb, ggml_tensor * dst, ثابت ggml_tensor * gate_scale_درون = nullptr);
فراغ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
bool ggml_vk_use_mul_mat_vec_id(ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);

// flash-attn
فراغ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * q, ثابت ggml_tensor * k, ثابت ggml_tensor * v, ثابت ggml_tensor * ماسک, ثابت ggml_tensor * sinks, ggml_tensor * dst);

// operators
فراغ ggml_vk_cpy_مقصد_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor, ثابت vk_subbuffer & درون, ثابت vk_subbuffer & بیرون);
bool ggml_vk_can_use_fwht(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);
فراغ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src, ggml_tensor * dst);
فراغ ggml_vk_get_ردیفها(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_get_ردیفها_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_multi_جمع_کن(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_جمع_کن(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_بیرون_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
صحيح ggml_vk_unary_mul_op_شاخص(ggml_unary_op op);
فراغ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_جمع_کن_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
فراغ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_gated_خطی_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_ssm_پیمایش(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_opt_گام_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_opt_گام_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
فراغ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_repeat_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_مجموعه_ردیفها(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_silu_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_گروه_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_اندازه(ggml_backend_vk_context * ctx, ثابت ggml_tensor *گره);
فراغ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, عائم * op_params);
فراغ ggml_vk_rms_norm_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_diag_ماسک_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
فراغ ggml_vk_soft_max_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح گره_idx, bool backprop);
فراغ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_جمع(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_جمع_ردیفها(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_میانگین(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_cross_entropy_loss_پشت(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_شمار_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);

//گراف
فراغ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_گراف(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح گره_idx, ggml_tensor *گره_begin, صحيح گره_idx_begin, bool آخرین_گره, bool almost_ready, bool submit);
فراغ ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, صحيح tensor_idx, bool almost_ready);
فراغ ggml_vk_گراف_cleanup(ggml_backend_vk_context * ctx);
فراغ ggml_vk_cleanup(ggml_backend_vk_context * ctx);
فراغ ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_خالی(ggml_tensor * گره);
bool ggml_vk_can_fuse(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, std::initializer_list<تعداد ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, صحيح رقم_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx, topk_moe_نما نما);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
bool ggml_vk_can_fuse_rope_مجموعه_ردیفها(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
bool ggml_vk_can_fuse_rms_norm_مجموعه_ردیفها(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
bool ggml_vk_tensors_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
uint32_t ggml_vk_fuse_multi_جمع_کن(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح گره_idx);
فراغ ggml_vk_گراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * گراف, بنية ggml_backend_گراف_optimize_params * params);

// backend
bool ggml_backend_میانگیر_is_vk(ggml_backend_میانگیر_t میانگیر);
فراغ ggml_backend_vk_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر);
فراغ * ggml_backend_vk_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر);
تعداد ggml_وضعیت ggml_backend_vk_میانگیر_init_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor);
فراغ ggml_backend_vk_میانگیر_memset_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, uint8_t مقدار, size_t جابه_جایی, size_t اندازه);
فراغ ggml_backend_vk_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t اندازه);
فراغ ggml_backend_vk_میانگیر_مجموعه_tensor_2d(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor, ثابت فراغ * داده, size_t جابه_جایی, size_t اندازه, size_t n_copies, size_t stride_tensor, size_t stride_داده);
فراغ ggml_backend_vk_میانگیر_get_tensor(ggml_backend_میانگیر_t میانگیر, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t اندازه);
فراغ ggml_backend_vk_میانگیر_get_tensor_2d(ggml_backend_میانگیر_t میانگیر, ثابت ggml_tensor * tensor, فراغ * داده, size_t جابه_جایی, size_t اندازه, size_t n_copies, size_t stride_tensor, size_t stride_داده);
bool ggml_backend_vk_میانگیر_cpy_tensor(ggml_backend_میانگیر_t میانگیر, ثابت ggml_tensor * src, ggml_tensor * dst);
فراغ ggml_backend_vk_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t مقدار);
ثابت حرف * ggml_backend_vk_میانگیر_type_name(ggml_backend_میانگیر_type_t buft);
ggml_backend_میانگیر_t ggml_backend_vk_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t اندازه);
size_t ggml_backend_vk_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft);
size_t ggml_backend_vk_میانگیر_type_get_max_اندازه(ggml_backend_میانگیر_type_t buft);
size_t ggml_backend_vk_میانگیر_type_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft, ثابت ggml_tensor * tensor);
فراغ ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_اندازه(ثابت ggml_tensor * op);
#اگر_تعریف_شده GGML_VULKAN_پیاپی_TESTS
قالب <typename X_TYPE, typename Y_TYPE>
فراغ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t رقم_it, صحيح تقسیم_k, صحيح shader_اندازه);
فراغ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
فراغ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t رقم_it, size_t تقسیم_k, size_t shader_اندازه, ggml_type quant, bool mmq = false);
#پایان_اگر
#اگر_تعریف_شده GGML_VULKAN_CHECK_RESULTS
خارجي size_t vk_skip_checks;
خارجي size_t vk_output_tensor;
فراغ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx);
فراغ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx);
#پایان_اگر

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
فراغ ggml_vk_print_device_fault_info(ثابت vk_device& device);
uint64_t ggml_vk_get_گره_flops(ثابت ggml_tensor * گره);
فراغ ggml_vk_print_گره_list(ثابت ggml_cgraph * cgraph, صحيح شروع, صحيح انتها);
فراغ ggml_vk_print_device_گمشده_info(ثابت vk_device& device);
size_t ggml_vk_tensor_میانگیر_جابه_جایی(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
size_t ggml_vk_descriptor_جابه_جایی(size_t tensor_جابه_جایی, size_t alignment, size_t type_اندازه);
uint32_t ggml_vk_concat_unit_اندازه(ggml_type type);
bool ggml_vk_concat_supported(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);

قالب <typename T>
مضمن فراغ ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_میانگیر_Info> ثابت& descriptor_میانگیر_infos, ثابت T &افزودن_constants, std::array<uint32_t, 3> elements) {
    ثابت uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ثابت uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ثابت uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    لكل (خودکار& میانگیر : descriptor_میانگیر_infos) {
        std::cerr << "(" << میانگیر.میانگیر << ", " << میانگیر.جابه_جایی << ", " << میانگیر.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_گروه_شمار[2]);
    GGML_ASSERT(ctx->descriptor_مجموعه_idx < ctx->descriptor_sets.اندازه());
    GGML_ASSERT(descriptor_میانگیر_infos.اندازه() <= MAX_PARAMETER_شمار);
    GGML_ASSERT(pipeline->parameter_شمار == descriptor_میانگیر_infos.اندازه());
    GGML_ASSERT(pipeline->افزودن_constant_اندازه == افزودن_constant_اندازه(افزودن_constants));

    ثابت uint32_t descriptor_مجموعه_idx = ctx->descriptor_مجموعه_idx++;
    vk::Descriptor_مجموعه& descriptor_مجموعه = ctx->descriptor_sets[descriptor_مجموعه_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ثابت uint64_t destroy_شمار = ctx->device->میانگیر_destroy_شمار.بارگیری(std::memory_order_acquire);
    إذا (ctx->descriptor_مجموعه_bindings_destroy_شمار != destroy_شمار) {
        لكل (خودکار & b : ctx->descriptor_مجموعه_bindings) {
            b.clear();
        }
        ctx->descriptor_مجموعه_bindings_destroy_شمار = destroy_شمار;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::وکتور<vk::Descriptor_میانگیر_Info> & bindings = ctx->descriptor_مجموعه_bindings[descriptor_مجموعه_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.اندازه() == descriptor_میانگیر_infos.اندازه();
    إذا (same) {
        size_t i = 0;
        لكل (ثابت vk::Descriptor_میانگیر_Info & info : descriptor_میانگیر_infos) {
            ثابت vk::Descriptor_میانگیر_Info & prev = bindings[i++];
            إذا (prev.میانگیر != info.میانگیر || prev.جابه_جایی != info.جابه_جایی || prev.range != info.range) {
                same = false;
                اكسر;
            }
        }
    }
    إذا (!same) {
        vk::Write_Descriptor_مجموعه write_descriptor_مجموعه{ descriptor_مجموعه, 0, 0, pipeline->parameter_شمار, vk::DescriptorType::e_Storage_میانگیر, nullptr, descriptor_میانگیر_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_مجموعه }, {});
        bindings.تخصیص_بده(descriptor_میانگیر_infos.begin(), descriptor_میانگیر_infos.انتها());
    }

    subctx->s->میانگیر->buf.افزودن_Constants(pipeline->layout, vk::Shader_Stage_Flag_بیتها::eCompute, 0, افزودن_constant_اندازه(افزودن_constants), افزودن_constant_داده(افزودن_constants));
    subctx->s->میانگیر->buf.bindPipeline(vk::Pipeline_Bind_نقطه::eCompute, pipeline->pipeline);
    subctx->s->میانگیر->buf.bindDescriptorSets(vk::Pipeline_Bind_نقطه::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_مجموعه },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->میانگیر->buf.dispatch(wg0, wg1, wg2);
    }
}

