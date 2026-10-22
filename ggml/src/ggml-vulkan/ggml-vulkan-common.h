#لارښوونه once
#ضمن "ggml-vulkan-push-constants.h"

مضمن std::ostream & چلوونکی<<(std::ostream & os, vk::بفر_2 بفر) {
    أرجع os << static_cast<Vk_بفر>(بفر);
}

// shared globals
خارجي ggml_backend_بفر_type_i ggml_backend_vk_بفر_type_interface;
خارجي bool vk_حافظه_logger_enabled;
خارجي bool vk_perf_logger_enabled;
خارجي bool vk_perf_logger_concurrent;
خارجي bool vk_enable_sync_logger;
خارجي uint32_t vk_perf_logger_frequency;
خارجي std::string vk_pipeline_stats_filter;
خارجي فراغ * ثابت vk_ptr_بنسټ;
خارجي vk_instance_t vk_instance;
خارجي ggml_backend_بفر_i ggml_backend_vk_بفر_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
فراغ ggml_vk_instance_init();
فراغ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
صحيح ggml_vk_get_device_شمېر();
فراغ ggml_vk_get_device_description(صحيح device, حرف * description, size_t description_اندازه);
bool ggml_vk_instance_layer_settings_شته();
bool ggml_vk_instance_portability_enumeration_ext_شته(ثابت std::ویکټور<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_شته(ثابت std::ویکټور<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ثابت vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_ماتریکس_support(ثابت vk::PhysicalDeviceProperties& props, ثابت vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_شمېر(ثابت vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_دننه_range(uint32_t driver_نسخه, uint32_t ښکتنی_major, uint32_t ښکتنی_minor, uint32_t پاسنی_major, uint32_t پاسنی_minor);

// shaders
فراغ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ثابت vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_لیکې, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_ماسک, bool use_ماسک_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_اندازه(ثابت std::string &pipeline_name, ثابت vk_device_architecture &arch);
فراغ ggml_vk_بارول_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ثابت vk_device& device, ثابت vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_بفر ggml_vk_create_بفر_check(vk_device& device, size_t اندازه, vk::حافظه_Property_Flags req_flags, vk::حافظه_Property_Flags fallback_flags = vk::حافظه_Property_Flags(0));
vk_بفر ggml_vk_create_بفر_device(vk_device& device, size_t اندازه);
فراغ ggml_vk_destroy_بفر(vk_بفر& buf);
فراغ * ggml_vk_host_malloc(size_t اندازه);
فراغ ggml_vk_host_free(فراغ * ptr);
فراغ ggml_vk_host_get(ثابت vk_device& device, ثابت فراغ * ptr, vk_بفر& buf, size_t& buf_دننه_واټن);
فراغ ggml_vk_ensure_sync_staging_بفر(vk_device& device, size_t اندازه);
فراغ ggml_vk_ensure_sync_staging_بفر(ggml_backend_vk_context * ctx, size_t اندازه);
bool ggml_vk_بفر_write_2d_async(vk_context subctx, vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی, bool sync_staging = false);
bool ggml_vk_بفر_write_async(vk_context subctx, vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t اندازه, bool sync_staging = false);
فراغ ggml_vk_بفر_write_2d(vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی);
فراغ ggml_vk_بفر_write(vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t اندازه);
bool ggml_vk_بفر_read_2d_async(vk_context subctx, vk_بفر& src, size_t دننه_واټن, فراغ * dst, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی, bool sync_staging = false);
فراغ ggml_vk_بفر_read_2d(vk_بفر& src, size_t دننه_واټن, فراغ * dst, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی);
فراغ ggml_vk_بفر_read(vk_بفر& src, size_t دننه_واټن, فراغ * dst, size_t اندازه);
فراغ ggml_vk_بفر_copy_async(vk_context& ctx, vk_بفر& dst, size_t dst_دننه_واټن, vk_بفر& src, size_t src_دننه_واټن, size_t اندازه);
فراغ ggml_vk_بفر_copy(vk_بفر& dst, size_t dst_دننه_واټن, vk_بفر& src, size_t src_دننه_واټن, size_t اندازه);
فراغ ggml_vk_بفر_memset_async(vk_context& ctx, vk_بفر& dst, size_t دننه_واټن, uint32_t c, size_t اندازه);
فراغ ggml_vk_بفر_memset(vk_بفر& dst, size_t دننه_واټن, uint32_t c, size_t اندازه);
vk_بفر ggml_vk_بفر_from_host_ptr(vk_device & device, فراغ * ptr, size_t اندازه);

// pipelines
uint64_t vk_tensor_دننه_واټن(ثابت ggml_tensor * tensor);
uint32_t get_misalign_بایټونه(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
فراغ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
فراغ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
فراغ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
فراغ ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_پیدا_کړه_کتار_family_شاخص(std::ویکټور<vk::کتار_Family_Properties>& کتار_family_props, ثابت vk::کتار_Flags& required, ثابت vk::کتار_Flags& avoid, int32_t compute_شاخص, uint32_t min_عدد_queues);
std::unique_ptr<vk_کتار> ggml_vk_create_کتار(vk_device& device, uint32_t کتار_family_شاخص, uint32_t کتار_شاخص, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_کتار> ggml_vk_create_aliased_کتار(vk_device& device, ثابت std::unique_ptr<vk_کتار>& سرچینه);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
فراغ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
فراغ ggml_vk_کتار_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ثابت ggml_backend_vk_context* ctx, ثابت vk_بفر& buf, size_t دننه_واټن = 0);
فراغ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
فراغ ggml_vk_سېټ_event(vk_context& ctx, vk::Event& event);
فراغ ggml_vk_wait_events(vk_context& ctx, std::ویکټور<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * tensor, bool allow_misalign = false);
فراغ ggml_vk_cmd_label_begin(vk::Command_بفر buf, ثابت حرف * name);
فراغ ggml_vk_ctx_پای_ټکی(vk_context& ctx);
فراغ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_اندازه(size_t پلنوالی, size_t align);
فراغ deferred_memcpy(فراغ * dst, ثابت فراغ * src, size_t اندازه, std::ویکټور<vk_staging_memcpy>* memcpys = nullptr);
فراغ deferred_memset(فراغ * dst, uint32_t val, size_t اندازه, std::ویکټور<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_منزل_fp16(ggml_backend_vk_context * ctx, ggml_type type);
فراغ ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& وېشل_k_بفر, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t وېشل_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_ستون_بایټونه, uint64_t d_ستون_بایټونه);
bool ggml_vk_dim01_contiguous(ثابت ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ثابت ggml_tensor * src, ثابت ggml_tensor * dst, ggml_type منزل);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
فراغ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت vk_subbuffer & دننه, ثابت vk_subbuffer & بهر, uint32_t ne);
فراغ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * mixes, ثابت ggml_tensor * scale, ثابت ggml_tensor * بنسټ, ggml_tensor * dst);
فراغ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * weights, ggml_tensor * dst);
فراغ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * x, ثابت ggml_tensor * residual, ثابت ggml_tensor * post, ثابت ggml_tensor * comb, ggml_tensor * dst, ثابت ggml_tensor * gate_scale_دننه = nullptr);
فراغ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
bool ggml_vk_use_mul_mat_vec_id(ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);

// flash-attn
فراغ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * q, ثابت ggml_tensor * k, ثابت ggml_tensor * v, ثابت ggml_tensor * ماسک, ثابت ggml_tensor * sinks, ggml_tensor * dst);

// operators
فراغ ggml_vk_cpy_منزل_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ثابت ggml_tensor * tensor, ثابت vk_subbuffer & دننه, ثابت vk_subbuffer & بهر);
bool ggml_vk_can_use_fwht(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);
فراغ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src, ggml_tensor * dst);
فراغ ggml_vk_get_لیکې(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_get_لیکې_شا(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_multi_جمع_کړه(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_جمع_کړه(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_بهر_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
صحيح ggml_vk_unary_mul_op_شاخص(ggml_unary_op op);
فراغ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_جمع_کړه_id(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
فراغ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_gated_خطي_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_ssm_لټون(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_opt_ګام_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_opt_ګام_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
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
فراغ ggml_vk_repeat_شا(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_سېټ_لیکې(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_silu_شا(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_ډله_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_اندازه(ggml_backend_vk_context * ctx, ثابت ggml_tensor *غوټه);
فراغ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, عائم * op_params);
فراغ ggml_vk_rms_norm_شا(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_diag_ماسک_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * src2, ggml_tensor * dst);
فراغ ggml_vk_soft_max_شا(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح غوټه_idx, bool backprop);
فراغ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_مجموعه(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_مجموعه_لیکې(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_منځنۍ(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_cross_entropy_loss_شا(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
فراغ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_شمېر_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);
فراغ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst);
فراغ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ثابت ggml_tensor * src0, ggml_tensor * dst);

//ګراف
فراغ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_ګراف(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح غوټه_idx, ggml_tensor *غوټه_begin, صحيح غوټه_idx_begin, bool وروستی_غوټه, bool almost_ready, bool submit);
فراغ ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, صحيح tensor_idx, bool almost_ready);
فراغ ggml_vk_ګراف_cleanup(ggml_backend_vk_context * ctx);
فراغ ggml_vk_cleanup(ggml_backend_vk_context * ctx);
فراغ ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_تشه(ggml_tensor * غوټه);
bool ggml_vk_can_fuse(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, std::initializer_list<تعداد ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ثابت ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, صحيح عدد_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, topk_moe_ډېرپېښ ډېرپېښ);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
bool ggml_vk_can_fuse_rope_سېټ_لیکې(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
bool ggml_vk_can_fuse_rms_norm_سېټ_لیکې(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
bool ggml_vk_tensors_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
uint32_t ggml_vk_fuse_multi_جمع_کړه(ggml_backend_vk_context * ctx, ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx);
فراغ ggml_vk_ګراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * ګراف, بنية ggml_backend_ګراف_optimize_params * params);

// backend
bool ggml_backend_بفر_is_vk(ggml_backend_بفر_t بفر);
فراغ ggml_backend_vk_بفر_free_بفر(ggml_backend_بفر_t بفر);
فراغ * ggml_backend_vk_بفر_get_بنسټ(ggml_backend_بفر_t بفر);
تعداد ggml_حالت ggml_backend_vk_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor);
فراغ ggml_backend_vk_بفر_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t ارزښت, size_t دننه_واټن, size_t اندازه);
فراغ ggml_backend_vk_بفر_سېټ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t اندازه);
فراغ ggml_backend_vk_بفر_سېټ_tensor_2d(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t دننه_واټن, size_t اندازه, size_t n_copies, size_t stride_tensor, size_t stride_معلومات);
فراغ ggml_backend_vk_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t اندازه);
فراغ ggml_backend_vk_بفر_get_tensor_2d(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t دننه_واټن, size_t اندازه, size_t n_copies, size_t stride_tensor, size_t stride_معلومات);
bool ggml_backend_vk_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst);
فراغ ggml_backend_vk_بفر_clear(ggml_backend_بفر_t بفر, uint8_t ارزښت);
ثابت حرف * ggml_backend_vk_بفر_type_name(ggml_backend_بفر_type_t buft);
ggml_backend_بفر_t ggml_backend_vk_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t اندازه);
size_t ggml_backend_vk_بفر_type_get_alignment(ggml_backend_بفر_type_t buft);
size_t ggml_backend_vk_بفر_type_get_max_اندازه(ggml_backend_بفر_type_t buft);
size_t ggml_backend_vk_بفر_type_get_alloc_اندازه(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor);
فراغ ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_اندازه(ثابت ggml_tensor * op);
#که_تعریف_شوی GGML_VULKAN_پرلپسې_TESTS
کينډۍ <typename X_TYPE, typename Y_TYPE>
فراغ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t عدد_it, صحيح وېشل_k, صحيح shader_اندازه);
فراغ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
فراغ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t عدد_it, size_t وېشل_k, size_t shader_اندازه, ggml_type quant, bool mmq = false);
#د_که_پای
#که_تعریف_شوی GGML_VULKAN_CHECK_RESULTS
خارجي size_t vk_skip_checks;
خارجي size_t vk_output_tensor;
فراغ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx);
فراغ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx);
#د_که_پای

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
فراغ ggml_vk_print_device_fault_info(ثابت vk_device& device);
uint64_t ggml_vk_get_غوټه_flops(ثابت ggml_tensor * غوټه);
فراغ ggml_vk_print_غوټه_list(ثابت ggml_cgraph * cgraph, صحيح پیل, صحيح پای_ټکی);
فراغ ggml_vk_print_device_ورک_info(ثابت vk_device& device);
size_t ggml_vk_tensor_بفر_دننه_واټن(ثابت ggml_backend_vk_context * ctx, ثابت ggml_tensor * t);
size_t ggml_vk_descriptor_دننه_واټن(size_t tensor_دننه_واټن, size_t alignment, size_t type_اندازه);
uint32_t ggml_vk_concat_unit_اندازه(ggml_type type);
bool ggml_vk_concat_supported(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ثابت ggml_tensor * dst);

کينډۍ <typename T>
مضمن فراغ ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_بفر_Info> ثابت& descriptor_بفر_infos, ثابت T &ټېل_constants, std::array<uint32_t, 3> elements) {
    ثابت uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ثابت uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ثابت uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    لكل (خپلکاری& بفر : descriptor_بفر_infos) {
        std::cerr << "(" << بفر.بفر << ", " << بفر.دننه_واټن << ", " << بفر.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_ډله_شمېر[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_ډله_شمېر[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_ډله_شمېر[2]);
    GGML_ASSERT(ctx->descriptor_سېټ_idx < ctx->descriptor_sets.اندازه());
    GGML_ASSERT(descriptor_بفر_infos.اندازه() <= MAX_PARAMETER_شمېر);
    GGML_ASSERT(pipeline->parameter_شمېر == descriptor_بفر_infos.اندازه());
    GGML_ASSERT(pipeline->ټېل_constant_اندازه == ټېل_constant_اندازه(ټېل_constants));

    ثابت uint32_t descriptor_سېټ_idx = ctx->descriptor_سېټ_idx++;
    vk::Descriptor_سېټ& descriptor_سېټ = ctx->descriptor_sets[descriptor_سېټ_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ثابت uint64_t destroy_شمېر = ctx->device->بفر_destroy_شمېر.بارول(std::memory_order_acquire);
    إذا (ctx->descriptor_سېټ_bindings_destroy_شمېر != destroy_شمېر) {
        لكل (خپلکاری & b : ctx->descriptor_سېټ_bindings) {
            b.clear();
        }
        ctx->descriptor_سېټ_bindings_destroy_شمېر = destroy_شمېر;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::ویکټور<vk::Descriptor_بفر_Info> & bindings = ctx->descriptor_سېټ_bindings[descriptor_سېټ_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.اندازه() == descriptor_بفر_infos.اندازه();
    إذا (same) {
        size_t i = 0;
        لكل (ثابت vk::Descriptor_بفر_Info & info : descriptor_بفر_infos) {
            ثابت vk::Descriptor_بفر_Info & prev = bindings[i++];
            إذا (prev.بفر != info.بفر || prev.دننه_واټن != info.دننه_واټن || prev.range != info.range) {
                same = false;
                اكسر;
            }
        }
    }
    إذا (!same) {
        vk::Write_Descriptor_سېټ write_descriptor_سېټ{ descriptor_سېټ, 0, 0, pipeline->parameter_شمېر, vk::DescriptorType::e_Storage_بفر, nullptr, descriptor_بفر_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_سېټ }, {});
        bindings.ټاکل(descriptor_بفر_infos.begin(), descriptor_بفر_infos.پای_ټکی());
    }

    subctx->s->بفر->buf.ټېل_Constants(pipeline->layout, vk::Shader_Stage_Flag_بټونه::eCompute, 0, ټېل_constant_اندازه(ټېل_constants), ټېل_constant_معلومات(ټېل_constants));
    subctx->s->بفر->buf.bindPipeline(vk::Pipeline_Bind_ټکی::eCompute, pipeline->pipeline);
    subctx->s->بفر->buf.bindDescriptorSets(vk::Pipeline_Bind_ټکی::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_سېټ },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->بفر->buf.dispatch(wg0, wg1, wg2);
    }
}

