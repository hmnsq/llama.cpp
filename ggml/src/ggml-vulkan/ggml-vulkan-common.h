#هدايت once
#ضمن "ggml-vulkan-push-constants.h"

مضمن std::ostream & عامل<<(std::ostream & os, vk::بفر_2 بفر) {
    أرجع os << static_cast<Vk_بفر>(بفر);
}

// shared globals
خارجي ggml_backend_بفر_type_i ggml_backend_vk_بفر_type_interface;
خارجي bool vk_يادگيري_logger_enabled;
خارجي bool vk_perf_logger_enabled;
خارجي bool vk_perf_logger_concurrent;
خارجي bool vk_enable_sync_logger;
خارجي uint32_t vk_perf_logger_frequency;
خارجي std::string vk_pipeline_stats_filter;
خارجي فراغ * ثابت vk_ptr_بنياد;
خارجي vk_instance_t vk_instance;
خارجي ggml_backend_بفر_i ggml_backend_vk_بفر_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
فراغ ggml_vk_instance_init();
فراغ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
صحيح ggml_vk_get_device_ڳڻپ();
فراغ ggml_vk_get_device_description(صحيح device, حرف * description, size_t description_سائيز);
bool ggml_vk_instance_layer_settings_موجود();
bool ggml_vk_instance_portability_enumeration_ext_موجود(ثابت std::ويڪٽر<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_موجود(ثابت std::ويڪٽر<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ثابت vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_ميٽرڪس_support(ثابت vk::PhysicalDeviceProperties& props, ثابت vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_ڳڻپ(ثابت vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_اندريون_range(uint32_t driver_نسخو, uint32_t هيٺاهون_major, uint32_t هيٺاهون_minor, uint32_t مٿانهون_major, uint32_t مٿانهون_minor);

// shaders
فراغ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_صفون, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_نقاب, bool use_نقاب_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_سائيز(ثابت std::string &pipeline_name, ثابت vk_device_architecture &arch);
فراغ ggml_vk_ڀريو_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_بفر ggml_vk_create_بفر_check(vk_device& device, size_t سائيز, vk::يادگيري_Property_Flags req_flags, vk::يادگيري_Property_Flags fallback_flags = vk::يادگيري_Property_Flags(0));
vk_بفر ggml_vk_create_بفر_device(vk_device& device, size_t سائيز);
فراغ ggml_vk_destroy_بفر(vk_بفر& buf);
فراغ * ggml_vk_host_malloc(size_t سائيز);
فراغ ggml_vk_host_free(فراغ * ptr);
فراغ ggml_vk_host_get(ثابت vk_device& device, ثابت فراغ * ptr, vk_بفر& buf, size_t& buf_وٿي);
فراغ ggml_vk_ensure_sync_staging_بفر(vk_device& device, size_t سائيز);
فراغ ggml_vk_ensure_sync_staging_بفر(ggml_backend_vk_context * ctx, size_t سائيز);
bool ggml_vk_بفر_write_2d_async(vk_context subctx, vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي, bool sync_staging = false);
bool ggml_vk_بفر_write_async(vk_context subctx, vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t سائيز, bool sync_staging = false);
فراغ ggml_vk_بفر_write_2d(vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي);
فراغ ggml_vk_بفر_write(vk_بفر& dst, size_t وٿي, ثابت فراغ * src, size_t سائيز);
bool ggml_vk_بفر_read_2d_async(vk_context subctx, vk_بفر& src, size_t وٿي, فراغ * dst, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي, bool sync_staging = false);
فراغ ggml_vk_بفر_read_2d(vk_بفر& src, size_t وٿي, فراغ * dst, size_t spitch, size_t dpitch, size_t ويڪر, size_t اوچائي);
فراغ ggml_vk_بفر_read(vk_بفر& src, size_t وٿي, فراغ * dst, size_t سائيز);
فراغ ggml_vk_بفر_copy_async(vk_context& ctx, vk_بفر& dst, size_t dst_وٿي, vk_بفر& src, size_t src_وٿي, size_t سائيز);
فراغ ggml_vk_بفر_copy(vk_بفر& dst, size_t dst_وٿي, vk_بفر& src, size_t src_وٿي, size_t سائيز);
فراغ ggml_vk_بفر_memset_async(vk_context& ctx, vk_بفر& dst, size_t وٿي, uint32_t c, size_t سائيز);
فراغ ggml_vk_بفر_memset(vk_بفر& dst, size_t وٿي, uint32_t c, size_t سائيز);
vk_بفر ggml_vk_بفر_from_host_ptr(vk_device & device, فراغ * ptr, size_t سائيز);

// pipelines
uint64_t vk_tensor_وٿي(ثابت ggml_tensor * tensor);
uint32_t get_misalign_بائيٽون(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
فراغ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
فراغ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
فراغ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
فراغ ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_ڳوليو_قطار_family_اشاريو(std::ويڪٽر<vk::قطار_Family_Properties>& قطار_family_props, ثابت vk::قطار_Flags& required, ثابت vk::قطار_Flags& avoid, int32_t compute_اشاريو, uint32_t min_انگ_queues);
std::unique_ptr<vk_قطار> ggml_vk_create_قطار(vk_device& device, uint32_t قطار_family_اشاريو, uint32_t قطار_اشاريو, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_قطار> ggml_vk_create_aliased_قطار(vk_device& device, ثابت std::unique_ptr<vk_قطار>& ذريعو);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
فراغ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
فراغ ggml_vk_قطار_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ثابت ggml_backend_vk_context* ctx, ثابت vk_بفر& buf, size_t وٿي = 0);
فراغ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
فراغ ggml_vk_سيٽ_event(vk_context& ctx, vk::Event& event);
فراغ ggml_vk_wait_events(vk_context& ctx, std::ويڪٽر<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * tensor, bool allow_misalign = false);
فراغ ggml_vk_cmd_label_begin(vk::Command_بفر buf, ثابت حرف * name);
فراغ ggml_vk_ctx_پڇاڙي(vk_context& ctx);
فراغ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_سائيز(size_t ويڪر, size_t align);
فراغ deferred_memcpy(فراغ * dst, ثابت فراغ * src, size_t سائيز, std::ويڪٽر<vk_staging_memcpy>* memcpys = nullptr);
فراغ deferred_memset(فراغ * dst, uint32_t val, size_t سائيز, std::ويڪٽر<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_منزل_fp16(ggml_backend_vk_context * ctx, ggml_type type);
فراغ ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ورهايو_k_بفر, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t ورهايو_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_کانو_بائيٽون, uint64_t d_کانو_بائيٽون);
bool ggml_vk_dim01_contiguous(ثابت ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src, ثابت ggml_tensor * dst, ggml_type منزل);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
فراغ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت vk_subbuffer & اندريون, ثابت vk_subbuffer & ٻاهريون, uint32_t ne);
فراغ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * mixes, ثابت ggml_tensor * scale, ثابت ggml_tensor * بنياد, ggml_tensor * dst);
فراغ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * weights, ggml_tensor * dst);
فراغ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * residual, ثابت ggml_tensor * post, ثابت ggml_tensor * comb, ggml_tensor * dst, ثابت ggml_tensor * gate_scale_اندريون = nullptr);
فراغ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
bool ggml_vk_use_mul_mat_vec_id(ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);

// flash-attn
فراغ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * q, ثابت ggml_tensor * k, ثابت ggml_tensor * v, ثابت ggml_tensor * نقاب, ثابت ggml_tensor * sinks, ggml_tensor * dst);

// operators
فراغ ggml_vk_cpy_منزل_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor, ثابت vk_subbuffer & اندريون, ثابت vk_subbuffer & ٻاهريون);
bool ggml_vk_can_use_fwht(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);
فراغ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src, ggml_tensor * dst);
فراغ ggml_vk_get_صفون(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_get_صفون_پٺ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_multi_جوڙيو(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_جوڙيو(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_ٻاهريون_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
صحيح ggml_vk_unary_mul_op_اشاريو(ggml_unary_op op);
فراغ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_جوڙيو_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
فراغ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_gated_لڪيري_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_ssm_جاچ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_opt_قدم_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_opt_قدم_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
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
فراغ ggml_vk_repeat_پٺ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_سيٽ_صفون(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_silu_پٺ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_گروهه_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_سائيز(ggml_backend_vk_context * ctx, ثابت ggml_tensor *ڳنڍ);
فراغ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx, عائم * op_params);
فراغ ggml_vk_rms_norm_پٺ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_diag_نقاب_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
فراغ ggml_vk_soft_max_پٺ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح ڳنڍ_idx, bool backprop);
فراغ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_جوڙ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_جوڙ_صفون(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_سراسري(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_cross_entropy_loss_پٺ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_ڳڻپ_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);

//گراف
فراغ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_گراف(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح ڳنڍ_idx, ggml_tensor *ڳنڍ_begin, صحيح ڳنڍ_idx_begin, bool آخري_ڳنڍ, bool almost_ready, bool submit);
فراغ ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, صحيح tensor_idx, bool almost_ready);
فراغ ggml_vk_گراف_cleanup(ggml_backend_vk_context * ctx);
فراغ ggml_vk_cleanup(ggml_backend_vk_context * ctx);
فراغ ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_خالي_جايون(ggml_tensor * ڳنڍ);
bool ggml_vk_can_fuse(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx, std::initializer_list<تعداد ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx, صحيح انگ_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx, topk_moe_گهڻائي گهڻائي);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
bool ggml_vk_can_fuse_rope_سيٽ_صفون(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
bool ggml_vk_can_fuse_rms_norm_سيٽ_صفون(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
bool ggml_vk_tensors_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
uint32_t ggml_vk_fuse_multi_جوڙيو(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح ڳنڍ_idx);
فراغ ggml_vk_گراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * گراف, بنية ggml_backend_گراف_optimize_params * params);

// backend
bool ggml_backend_بفر_is_vk(ggml_backend_بفر_t بفر);
فراغ ggml_backend_vk_بفر_free_بفر(ggml_backend_بفر_t بفر);
فراغ * ggml_backend_vk_بفر_get_بنياد(ggml_backend_بفر_t بفر);
تعداد ggml_حالت ggml_backend_vk_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor);
فراغ ggml_backend_vk_بفر_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t قدر, size_t وٿي, size_t سائيز);
فراغ ggml_backend_vk_بفر_سيٽ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t وٿي, size_t سائيز);
فراغ ggml_backend_vk_بفر_سيٽ_tensor_2d(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t وٿي, size_t سائيز, size_t n_copies, size_t stride_tensor, size_t stride_معلومات);
فراغ ggml_backend_vk_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t وٿي, size_t سائيز);
فراغ ggml_backend_vk_بفر_get_tensor_2d(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t وٿي, size_t سائيز, size_t n_copies, size_t stride_tensor, size_t stride_معلومات);
bool ggml_backend_vk_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst);
فراغ ggml_backend_vk_بفر_clear(ggml_backend_بفر_t بفر, uint8_t قدر);
ثابت حرف * ggml_backend_vk_بفر_type_name(ggml_backend_بفر_type_t buft);
ggml_backend_بفر_t ggml_backend_vk_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t سائيز);
size_t ggml_backend_vk_بفر_type_get_alignment(ggml_backend_بفر_type_t buft);
size_t ggml_backend_vk_بفر_type_get_max_سائيز(ggml_backend_بفر_type_t buft);
size_t ggml_backend_vk_بفر_type_get_alloc_سائيز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor);
فراغ ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_سائيز(ثابت ggml_tensor * op);
#جيڪڏهن_وصف_ٿيل GGML_VULKAN_لاڳيتو_TESTS
سانچو <typename X_TYPE, typename Y_TYPE>
فراغ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t انگ_it, صحيح ورهايو_k, صحيح shader_سائيز);
فراغ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
فراغ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t انگ_it, size_t ورهايو_k, size_t shader_سائيز, ggml_type quant, bool mmq = false);
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_VULKAN_CHECK_RESULTS
خارجي size_t vk_skip_checks;
خارجي size_t vk_output_tensor;
فراغ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx);
فراغ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx);
#جيڪڏهن_پورو

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
فراغ ggml_vk_print_device_fault_info(ثابت vk_device& device);
uint64_t ggml_vk_get_ڳنڍ_flops(ثابت ggml_tensor * ڳنڍ);
فراغ ggml_vk_print_ڳنڍ_list(ثابت ggml_cgraph * cgraph, صحيح شروعات, صحيح پڇاڙي);
فراغ ggml_vk_print_device_گم_info(ثابت vk_device& device);
size_t ggml_vk_tensor_بفر_وٿي(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
size_t ggml_vk_descriptor_وٿي(size_t tensor_وٿي, size_t alignment, size_t type_سائيز);
uint32_t ggml_vk_concat_unit_سائيز(ggml_type type);
bool ggml_vk_concat_supported(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);

سانچو <typename T>
مضمن فراغ ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_بفر_Info> ثابت& descriptor_بفر_infos, ثابت T &ڌڪو_constants, std::array<uint32_t, 3> elements) {
    ثابت uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ثابت uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ثابت uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    لكل (خودڪار& بفر : descriptor_بفر_infos) {
        std::cerr << "(" << بفر.بفر << ", " << بفر.وٿي << ", " << بفر.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_گروهه_ڳڻپ[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_گروهه_ڳڻپ[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_گروهه_ڳڻپ[2]);
    GGML_ASSERT(ctx->descriptor_سيٽ_idx < ctx->descriptor_sets.سائيز());
    GGML_ASSERT(descriptor_بفر_infos.سائيز() <= MAX_PARAMETER_ڳڻپ);
    GGML_ASSERT(pipeline->parameter_ڳڻپ == descriptor_بفر_infos.سائيز());
    GGML_ASSERT(pipeline->ڌڪو_constant_سائيز == ڌڪو_constant_سائيز(ڌڪو_constants));

    ثابت uint32_t descriptor_سيٽ_idx = ctx->descriptor_سيٽ_idx++;
    vk::Descriptor_سيٽ& descriptor_سيٽ = ctx->descriptor_sets[descriptor_سيٽ_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ثابت uint64_t destroy_ڳڻپ = ctx->device->بفر_destroy_ڳڻپ.ڀريو(std::memory_order_acquire);
    إذا (ctx->descriptor_سيٽ_bindings_destroy_ڳڻپ != destroy_ڳڻپ) {
        لكل (خودڪار & b : ctx->descriptor_سيٽ_bindings) {
            b.clear();
        }
        ctx->descriptor_سيٽ_bindings_destroy_ڳڻپ = destroy_ڳڻپ;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::ويڪٽر<vk::Descriptor_بفر_Info> & bindings = ctx->descriptor_سيٽ_bindings[descriptor_سيٽ_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.سائيز() == descriptor_بفر_infos.سائيز();
    إذا (same) {
        size_t i = 0;
        لكل (ثابت vk::Descriptor_بفر_Info & info : descriptor_بفر_infos) {
            ثابت vk::Descriptor_بفر_Info & prev = bindings[i++];
            إذا (prev.بفر != info.بفر || prev.وٿي != info.وٿي || prev.range != info.range) {
                same = false;
                اكسر;
            }
        }
    }
    إذا (!same) {
        vk::Write_Descriptor_سيٽ write_descriptor_سيٽ{ descriptor_سيٽ, 0, 0, pipeline->parameter_ڳڻپ, vk::DescriptorType::e_Storage_بفر, nullptr, descriptor_بفر_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_سيٽ }, {});
        bindings.سونپيو(descriptor_بفر_infos.begin(), descriptor_بفر_infos.پڇاڙي());
    }

    subctx->s->بفر->buf.ڌڪو_Constants(pipeline->layout, vk::Shader_Stage_Flag_بٽ::eCompute, 0, ڌڪو_constant_سائيز(ڌڪو_constants), ڌڪو_constant_معلومات(ڌڪو_constants));
    subctx->s->بفر->buf.bindPipeline(vk::Pipeline_Bind_نقطو::eCompute, pipeline->pipeline);
    subctx->s->بفر->buf.bindDescriptorSets(vk::Pipeline_Bind_نقطو::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_سيٽ },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->بفر->buf.dispatch(wg0, wg1, wg2);
    }
}

