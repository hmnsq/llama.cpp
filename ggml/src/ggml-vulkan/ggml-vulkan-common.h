#निर्देशः once
#अन्तर्भाव "ggml-vulkan-push-constants.h"

अन्तःस्थ std::ostream & संक्रियकः<<(std::ostream & os, vk::बफर_2 बफर) {
    प्रत्यागम os << static_cast<Vk_बफर>(बफर);
}

// shared globals
बाह्य ggml_backend_बफर_type_i ggml_backend_vk_बफर_type_interface;
बाह्य bool vk_स्मृतिः_logger_enabled;
बाह्य bool vk_perf_logger_enabled;
बाह्य bool vk_perf_logger_concurrent;
बाह्य bool vk_enable_sync_logger;
बाह्य uint32_t vk_perf_logger_frequency;
बाह्य std::string vk_pipeline_stats_filter;
बाह्य शून्य * ध्रुव vk_ptr_आधारः;
बाह्य vk_instance_t vk_instance;
बाह्य ggml_backend_बफर_i ggml_backend_vk_बफर_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
शून्य ggml_vk_instance_init();
शून्य ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
पूर्णाङ्क ggml_vk_get_device_गणनम्();
शून्य ggml_vk_get_device_description(पूर्णाङ्क device, वर्ण * description, size_t description_आकार);
bool ggml_vk_instance_layer_settings_उपलब्धम्();
bool ggml_vk_instance_portability_enumeration_ext_उपलब्धम्(ध्रुव std::सदिशम्<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_उपलब्धम्(ध्रुव std::सदिशम्<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ध्रुव vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_matrix_support(ध्रुव vk::PhysicalDeviceProperties& props, ध्रुव vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_गणनम्(ध्रुव vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_अन्तर्_range(uint32_t driver_संस्करणम्, uint32_t अधस्तनम्_major, uint32_t अधस्तनम्_minor, uint32_t ऊर्ध्वस्थम्_major, uint32_t ऊर्ध्वस्थम्_minor);

// shaders
शून्य ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ध्रुव vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_पङ्क्तिकाः, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ध्रुव vk_device& device, ध्रुव vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_आवरणम्, bool use_आवरणम्_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_आकार(ध्रुव std::string &pipeline_name, ध्रुव vk_device_architecture &arch);
शून्य ggml_vk_आरोपय_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ध्रुव vk_device& device, ध्रुव vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ध्रुव vk_device& device, ध्रुव vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_बफर ggml_vk_create_बफर_check(vk_device& device, size_t आकार, vk::स्मृतिः_Property_Flags req_flags, vk::स्मृतिः_Property_Flags fallback_flags = vk::स्मृतिः_Property_Flags(0));
vk_बफर ggml_vk_create_बफर_device(vk_device& device, size_t आकार);
शून्य ggml_vk_destroy_बफर(vk_बफर& buf);
शून्य * ggml_vk_host_malloc(size_t आकार);
शून्य ggml_vk_host_free(शून्य * ptr);
शून्य ggml_vk_host_get(ध्रुव vk_device& device, ध्रुव शून्य * ptr, vk_बफर& buf, size_t& buf_विस्थापनम्);
शून्य ggml_vk_ensure_sync_staging_बफर(vk_device& device, size_t आकार);
शून्य ggml_vk_ensure_sync_staging_बफर(ggml_backend_vk_context * ctx, size_t आकार);
bool ggml_vk_बफर_write_2d_async(vk_context subctx, vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height, bool sync_staging = false);
bool ggml_vk_बफर_write_async(vk_context subctx, vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t आकार, bool sync_staging = false);
शून्य ggml_vk_बफर_write_2d(vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height);
शून्य ggml_vk_बफर_write(vk_बफर& dst, size_t विस्थापनम्, ध्रुव शून्य * src, size_t आकार);
bool ggml_vk_बफर_read_2d_async(vk_context subctx, vk_बफर& src, size_t विस्थापनम्, शून्य * dst, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height, bool sync_staging = false);
शून्य ggml_vk_बफर_read_2d(vk_बफर& src, size_t विस्थापनम्, शून्य * dst, size_t spitch, size_t dpitch, size_t विस्तारः, size_t height);
शून्य ggml_vk_बफर_read(vk_बफर& src, size_t विस्थापनम्, शून्य * dst, size_t आकार);
शून्य ggml_vk_बफर_copy_async(vk_context& ctx, vk_बफर& dst, size_t dst_विस्थापनम्, vk_बफर& src, size_t src_विस्थापनम्, size_t आकार);
शून्य ggml_vk_बफर_copy(vk_बफर& dst, size_t dst_विस्थापनम्, vk_बफर& src, size_t src_विस्थापनम्, size_t आकार);
शून्य ggml_vk_बफर_memset_async(vk_context& ctx, vk_बफर& dst, size_t विस्थापनम्, uint32_t c, size_t आकार);
शून्य ggml_vk_बफर_memset(vk_बफर& dst, size_t विस्थापनम्, uint32_t c, size_t आकार);
vk_बफर ggml_vk_बफर_from_host_ptr(vk_device & device, शून्य * ptr, size_t आकार);

// pipelines
uint64_t vk_tensor_विस्थापनम्(ध्रुव ggml_tensor * tensor);
uint32_t get_misalign_अष्टकानि(ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * t);
शून्य ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
शून्य ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
शून्य ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
शून्य ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_find_queue_family_सूचकांक(std::सदिशम्<vk::QueueFamilyProperties>& queue_family_props, ध्रुव vk::QueueFlags& required, ध्रुव vk::QueueFlags& avoid, int32_t compute_सूचकांक, uint32_t min_अङ्कः_queues);
std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_सूचकांक, uint32_t queue_सूचकांक, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, ध्रुव std::unique_ptr<vk_queue>& source);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
शून्य ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
शून्य ggml_vk_queue_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ध्रुव ggml_backend_vk_context* ctx, ध्रुव vk_बफर& buf, size_t विस्थापनम् = 0);
शून्य ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
शून्य ggml_vk_समुच्चयः_event(vk_context& ctx, vk::Event& event);
शून्य ggml_vk_wait_events(vk_context& ctx, std::सदिशम्<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * tensor, bool allow_misalign = false);
शून्य ggml_vk_cmd_label_begin(vk::Command_बफर buf, ध्रुव वर्ण * name);
शून्य ggml_vk_ctx_अन्तः(vk_context& ctx);
शून्य ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_आकार(size_t विस्तारः, size_t align);
शून्य deferred_memcpy(शून्य * dst, ध्रुव शून्य * src, size_t आकार, std::सदिशम्<vk_staging_memcpy>* memcpys = nullptr);
शून्य deferred_memset(शून्य * dst, uint32_t val, size_t आकार, std::सदिशम्<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_गन्तव्यम्_fp16(ggml_backend_vk_context * ctx, ggml_type type);
शून्य ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& विभज_k_बफर, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t विभज_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_col_अष्टकानि, uint64_t d_col_अष्टकानि);
bool ggml_vk_dim01_contiguous(ध्रुव ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * src, ध्रुव ggml_tensor * dst, ggml_type गन्तव्यम्);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
शून्य ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव vk_subbuffer & अन्तर्, ध्रुव vk_subbuffer & बहिर्गमनम्, uint32_t ne);
शून्य ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * mixes, ध्रुव ggml_tensor * scale, ध्रुव ggml_tensor * आधारः, ggml_tensor * dst);
शून्य ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * x, ध्रुव ggml_tensor * weights, ggml_tensor * dst);
शून्य ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * x, ध्रुव ggml_tensor * residual, ध्रुव ggml_tensor * post, ध्रुव ggml_tensor * comb, ggml_tensor * dst, ध्रुव ggml_tensor * gate_scale_अन्तर् = nullptr);
शून्य ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
bool ggml_vk_use_mul_mat_vec_id(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);

// flash-attn
शून्य ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * q, ध्रुव ggml_tensor * k, ध्रुव ggml_tensor * v, ध्रुव ggml_tensor * आवरणम्, ध्रुव ggml_tensor * sinks, ggml_tensor * dst);

// operators
शून्य ggml_vk_cpy_गन्तव्यम्_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ध्रुव ggml_tensor * tensor, ध्रुव vk_subbuffer & अन्तर्, ध्रुव vk_subbuffer & बहिर्गमनम्);
bool ggml_vk_can_use_fwht(ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * dst);
शून्य ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src, ggml_tensor * dst);
शून्य ggml_vk_get_पङ्क्तिकाः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_get_पङ्क्तिकाः_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_multi_सङ्कलय(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_सङ्कलय(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_बहिर्गमनम्_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
पूर्णाङ्क ggml_vk_unary_mul_op_सूचकांक(ggml_unary_op op);
शून्य ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_सङ्कलय_id(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ggml_tensor * dst);
शून्य ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_gated_रैखिकम्_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_ssm_अवलोकनम्(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_opt_चरणः_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_opt_चरणः_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ggml_tensor * dst);
शून्य ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_repeat_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_समुच्चयः_पङ्क्तिकाः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_silu_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_गणः_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_आकार(ggml_backend_vk_context * ctx, ध्रुव ggml_tensor *ग्रन्थिः);
शून्य ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, भिन्न * op_params);
शून्य ggml_vk_rms_norm_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_diag_आवरणम्_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * src2, ggml_tensor * dst);
शून्य ggml_vk_soft_max_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, bool backprop);
शून्य ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_योग(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_योग_पङ्क्तिकाः(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_माध्यम्(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_cross_entropy_loss_पृष्ठभागः(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
शून्य ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_गणनम्_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);
शून्य ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst);
शून्य ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ध्रुव ggml_tensor * src0, ggml_tensor * dst);

// graph
शून्य ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, ggml_tensor *ग्रन्थिः_begin, पूर्णाङ्क ग्रन्थिः_idx_begin, bool चरमः_ग्रन्थिः, bool almost_ready, bool submit);
शून्य ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, पूर्णाङ्क tensor_idx, bool almost_ready);
शून्य ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx);
शून्य ggml_vk_cleanup(ggml_backend_vk_context * ctx);
शून्य ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_रिक्तानि(ggml_tensor * ग्रन्थिः);
bool ggml_vk_can_fuse(ध्रुव ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, std::initializer_list<गणना ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ध्रुव ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, पूर्णाङ्क अङ्कः_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, topk_moe_बहुलकम् बहुलकम्);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
bool ggml_vk_can_fuse_rope_समुच्चयः_पङ्क्तिकाः(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
bool ggml_vk_can_fuse_rms_norm_समुच्चयः_पङ्क्तिकाः(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
bool ggml_vk_tensors_overlap(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
uint32_t ggml_vk_fuse_multi_सङ्कलय(ggml_backend_vk_context * ctx, ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx);
शून्य ggml_vk_graph_optimize(ggml_backend_t backend, संरचना ggml_cgraph * graph, संरचना ggml_backend_graph_optimize_params * params);

// backend
bool ggml_backend_बफर_is_vk(ggml_backend_बफर_t बफर);
शून्य ggml_backend_vk_बफर_free_बफर(ggml_backend_बफर_t बफर);
शून्य * ggml_backend_vk_बफर_get_आधारः(ggml_backend_बफर_t बफर);
गणना ggml_स्थितिः ggml_backend_vk_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor);
शून्य ggml_backend_vk_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t मानम्, size_t विस्थापनम्, size_t आकार);
शून्य ggml_backend_vk_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t आकार);
शून्य ggml_backend_vk_बफर_समुच्चयः_tensor_2d(ggml_backend_बफर_t बफर, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t आकार, size_t n_copies, size_t stride_tensor, size_t stride_दत्त);
शून्य ggml_backend_vk_बफर_get_tensor(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t आकार);
शून्य ggml_backend_vk_बफर_get_tensor_2d(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t आकार, size_t n_copies, size_t stride_tensor, size_t stride_दत्त);
bool ggml_backend_vk_बफर_cpy_tensor(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * src, ggml_tensor * dst);
शून्य ggml_backend_vk_बफर_clear(ggml_backend_बफर_t बफर, uint8_t मानम्);
ध्रुव वर्ण * ggml_backend_vk_बफर_type_name(ggml_backend_बफर_type_t buft);
ggml_backend_बफर_t ggml_backend_vk_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t आकार);
size_t ggml_backend_vk_बफर_type_get_alignment(ggml_backend_बफर_type_t buft);
size_t ggml_backend_vk_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft);
size_t ggml_backend_vk_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, ध्रुव ggml_tensor * tensor);
शून्य ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_आकार(ध्रुव ggml_tensor * op);
#यदि_परिभाषितम् GGML_VULKAN_सततम्_TESTS
प्रतिरूपम् <typename X_TYPE, typename Y_TYPE>
शून्य ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t अङ्कः_it, पूर्णाङ्क विभज_k, पूर्णाङ्क shader_आकार);
शून्य ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
शून्य ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t अङ्कः_it, size_t विभज_k, size_t shader_आकार, ggml_type quant, bool mmq = false);
#यदि_अन्तः
#यदि_परिभाषितम् GGML_VULKAN_CHECK_RESULTS
बाह्य size_t vk_skip_checks;
बाह्य size_t vk_output_tensor;
शून्य ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णाङ्क tensor_idx);
शून्य ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णाङ्क tensor_idx);
#यदि_अन्तः

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
शून्य ggml_vk_print_device_fault_info(ध्रुव vk_device& device);
uint64_t ggml_vk_get_ग्रन्थिः_flops(ध्रुव ggml_tensor * ग्रन्थिः);
शून्य ggml_vk_print_ग्रन्थिः_list(ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क start, पूर्णाङ्क अन्तः_2);
शून्य ggml_vk_print_device_नष्टम्_info(ध्रुव vk_device& device);
size_t ggml_vk_tensor_बफर_विस्थापनम्(ध्रुव ggml_backend_vk_context * ctx, ध्रुव ggml_tensor * t);
size_t ggml_vk_descriptor_विस्थापनम्(size_t tensor_विस्थापनम्, size_t alignment, size_t type_आकार);
uint32_t ggml_vk_concat_unit_आकार(ggml_type type);
bool ggml_vk_concat_supported(ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * dst);

प्रतिरूपम् <typename T>
अन्तःस्थ शून्य ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_बफर_Info> ध्रुव& descriptor_बफर_infos, ध्रुव T &प्रक्षिप_constants, std::array<uint32_t, 3> elements) {
    ध्रुव uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ध्रुव uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ध्रुव uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    प्रति (स्वचालित& बफर : descriptor_बफर_infos) {
        std::cerr << "(" << बफर.बफर << ", " << बफर.विस्थापनम् << ", " << बफर.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_गणः_गणनम्[2]);
    GGML_ASSERT(ctx->descriptor_समुच्चयः_idx < ctx->descriptor_sets.आकार());
    GGML_ASSERT(descriptor_बफर_infos.आकार() <= MAX_PARAMETER_गणनम्);
    GGML_ASSERT(pipeline->parameter_गणनम् == descriptor_बफर_infos.आकार());
    GGML_ASSERT(pipeline->प्रक्षिप_constant_आकार == प्रक्षिप_constant_आकार(प्रक्षिप_constants));

    ध्रुव uint32_t descriptor_समुच्चयः_idx = ctx->descriptor_समुच्चयः_idx++;
    vk::Descriptor_समुच्चयः& descriptor_समुच्चयः = ctx->descriptor_sets[descriptor_समुच्चयः_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ध्रुव uint64_t destroy_गणनम् = ctx->device->बफर_destroy_गणनम्.आरोपय(std::memory_order_acquire);
    यदि (ctx->descriptor_समुच्चयः_bindings_destroy_गणनम् != destroy_गणनम्) {
        प्रति (स्वचालित & b : ctx->descriptor_समुच्चयः_bindings) {
            b.clear();
        }
        ctx->descriptor_समुच्चयः_bindings_destroy_गणनम् = destroy_गणनम्;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::सदिशम्<vk::Descriptor_बफर_Info> & bindings = ctx->descriptor_समुच्चयः_bindings[descriptor_समुच्चयः_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.आकार() == descriptor_बफर_infos.आकार();
    यदि (same) {
        size_t i = 0;
        प्रति (ध्रुव vk::Descriptor_बफर_Info & info : descriptor_बफर_infos) {
            ध्रुव vk::Descriptor_बफर_Info & पूर्व = bindings[i++];
            यदि (पूर्व.बफर != info.बफर || पूर्व.विस्थापनम् != info.विस्थापनम् || पूर्व.range != info.range) {
                same = false;
                भङ्ग;
            }
        }
    }
    यदि (!same) {
        vk::Write_Descriptor_समुच्चयः write_descriptor_समुच्चयः{ descriptor_समुच्चयः, 0, 0, pipeline->parameter_गणनम्, vk::DescriptorType::e_Storage_बफर, nullptr, descriptor_बफर_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_समुच्चयः }, {});
        bindings.नियोजय(descriptor_बफर_infos.begin(), descriptor_बफर_infos.अन्तः_2());
    }

    subctx->s->बफर->buf.प्रक्षिप_Constants(pipeline->layout, vk::Shader_Stage_Flag_द्वयङ्काः::eCompute, 0, प्रक्षिप_constant_आकार(प्रक्षिप_constants), प्रक्षिप_constant_दत्त(प्रक्षिप_constants));
    subctx->s->बफर->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->बफर->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_समुच्चयः },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->बफर->buf.dispatch(wg0, wg1, wg2);
    }
}

