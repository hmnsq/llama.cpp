#आशय once
#समावेश "ggml-vulkan-push-constants.h"

अंतरभूत std::ostream & चालक<<(std::ostream & os, vk::बफर_2 बफर) {
    वापस os << static_cast<Vk_बफर>(बफर);
}

// shared globals
बाह्य ggml_backend_बफर_type_i ggml_backend_vk_बफर_type_interface;
बाह्य bool vk_memory_logger_enabled;
बाह्य bool vk_perf_logger_enabled;
बाह्य bool vk_perf_logger_concurrent;
बाह्य bool vk_enable_sync_logger;
बाह्य uint32_t vk_perf_logger_frequency;
बाह्य std::string vk_pipeline_stats_filter;
बाह्य व्योम * स्थिर vk_ptr_base;
बाह्य vk_instance_t vk_instance;
बाह्य ggml_backend_बफर_i ggml_backend_vk_बफर_interface;

// instance
vk_device ggml_vk_get_device(माप_प्रकार idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
व्योम ggml_vk_instance_init();
व्योम ggml_vk_init(ggml_backend_vk_context * ctx, माप_प्रकार idx);
पूर्णांक ggml_vk_get_device_गिनती();
व्योम ggml_vk_get_device_description(पूर्णांक device, अक्षर * description, माप_प्रकार description_आकार);
bool ggml_vk_instance_layer_settings_available();
bool ggml_vk_instance_portability_enumeration_ext_available(स्थिर std::vector<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_available(स्थिर std::vector<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(स्थिर vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_matrix_support(स्थिर vk::PhysicalDeviceProperties& props, स्थिर vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_गिनती(स्थिर vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_in_range(uint32_t driver_version, uint32_t lower_major, uint32_t lower_minor, uint32_t upper_major, uint32_t upper_minor);

// shaders
व्योम ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(स्थिर vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(स्थिर vk_device& device, स्थिर vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_mask, bool use_mask_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_आकार(स्थिर std::string &pipeline_name, स्थिर vk_device_architecture &arch);
व्योम ggml_vk_load_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(स्थिर vk_device& device, स्थिर vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(स्थिर vk_device& device, स्थिर vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_बफर ggml_vk_create_बफर_check(vk_device& device, माप_प्रकार आकार, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0));
vk_बफर ggml_vk_create_बफर_device(vk_device& device, माप_प्रकार आकार);
व्योम ggml_vk_destroy_बफर(vk_बफर& buf);
व्योम * ggml_vk_host_malloc(माप_प्रकार आकार);
व्योम ggml_vk_host_free(व्योम * ptr);
व्योम ggml_vk_host_get(स्थिर vk_device& device, स्थिर व्योम * ptr, vk_बफर& buf, माप_प्रकार& buf_offset);
व्योम ggml_vk_ensure_sync_staging_बफर(vk_device& device, माप_प्रकार आकार);
व्योम ggml_vk_ensure_sync_staging_बफर(ggml_backend_vk_context * ctx, माप_प्रकार आकार);
bool ggml_vk_बफर_write_2d_async(vk_context subctx, vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height, bool sync_staging = false);
bool ggml_vk_बफर_write_async(vk_context subctx, vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार आकार, bool sync_staging = false);
व्योम ggml_vk_बफर_write_2d(vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height);
व्योम ggml_vk_बफर_write(vk_बफर& dst, माप_प्रकार offset, स्थिर व्योम * src, माप_प्रकार आकार);
bool ggml_vk_बफर_read_2d_async(vk_context subctx, vk_बफर& src, माप_प्रकार offset, व्योम * dst, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height, bool sync_staging = false);
व्योम ggml_vk_बफर_read_2d(vk_बफर& src, माप_प्रकार offset, व्योम * dst, माप_प्रकार spitch, माप_प्रकार dpitch, माप_प्रकार width, माप_प्रकार height);
व्योम ggml_vk_बफर_read(vk_बफर& src, माप_प्रकार offset, व्योम * dst, माप_प्रकार आकार);
व्योम ggml_vk_बफर_copy_async(vk_context& ctx, vk_बफर& dst, माप_प्रकार dst_offset, vk_बफर& src, माप_प्रकार src_offset, माप_प्रकार आकार);
व्योम ggml_vk_बफर_copy(vk_बफर& dst, माप_प्रकार dst_offset, vk_बफर& src, माप_प्रकार src_offset, माप_प्रकार आकार);
व्योम ggml_vk_बफर_memset_async(vk_context& ctx, vk_बफर& dst, माप_प्रकार offset, uint32_t c, माप_प्रकार आकार);
व्योम ggml_vk_बफर_memset(vk_बफर& dst, माप_प्रकार offset, uint32_t c, माप_प्रकार आकार);
vk_बफर ggml_vk_बफर_from_host_ptr(vk_device & device, व्योम * ptr, माप_प्रकार आकार);

// pipelines
uint64_t vk_tensor_offset(स्थिर ggml_tensor * tensor);
uint32_t get_misalign_bytes(स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * t);
व्योम ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
व्योम ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
व्योम ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
व्योम ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_find_queue_family_सूचकांक(std::vector<vk::QueueFamilyProperties>& queue_family_props, स्थिर vk::QueueFlags& required, स्थिर vk::QueueFlags& avoid, int32_t compute_सूचकांक, uint32_t min_अंक_queues);
std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_सूचकांक, uint32_t queue_सूचकांक, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, स्थिर std::unique_ptr<vk_queue>& source);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
व्योम ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
व्योम ggml_vk_queue_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(स्थिर ggml_backend_vk_context* ctx, स्थिर vk_बफर& buf, माप_प्रकार offset = 0);
व्योम ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
व्योम ggml_vk_set_event(vk_context& ctx, vk::Event& event);
व्योम ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * tensor, bool allow_misalign = false);
व्योम ggml_vk_cmd_label_begin(vk::Command_बफर buf, स्थिर अक्षर * name);
व्योम ggml_vk_ctx_end(vk_context& ctx);
व्योम ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
माप_प्रकार ggml_vk_align_आकार(माप_प्रकार width, माप_प्रकार align);
व्योम deferred_memcpy(व्योम * dst, स्थिर व्योम * src, माप_प्रकार आकार, std::vector<vk_staging_memcpy>* memcpys = nullptr);
व्योम deferred_memset(व्योम * dst, uint32_t val, माप_प्रकार आकार, std::vector<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type);
व्योम ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_बफर, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_col_bytes, uint64_t d_col_bytes);
bool ggml_vk_dim01_contiguous(स्थिर ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, स्थिर ggml_tensor * src, स्थिर ggml_tensor * dst, ggml_type to);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
व्योम ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर vk_subbuffer & in, स्थिर vk_subbuffer & out, uint32_t ne);
व्योम ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * mixes, स्थिर ggml_tensor * scale, स्थिर ggml_tensor * base, ggml_tensor * dst);
व्योम ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * x, स्थिर ggml_tensor * weights, ggml_tensor * dst);
व्योम ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * x, स्थिर ggml_tensor * residual, स्थिर ggml_tensor * post, स्थिर ggml_tensor * comb, ggml_tensor * dst, स्थिर ggml_tensor * gate_scale_in = nullptr);
व्योम ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
bool ggml_vk_use_mul_mat_vec_id(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);

// flash-attn
व्योम ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * q, स्थिर ggml_tensor * k, स्थिर ggml_tensor * v, स्थिर ggml_tensor * mask, स्थिर ggml_tensor * sinks, ggml_tensor * dst);

// operators
व्योम ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, स्थिर ggml_tensor * tensor, स्थिर vk_subbuffer & in, स्थिर vk_subbuffer & out);
bool ggml_vk_can_use_fwht(स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * dst);
व्योम ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src, ggml_tensor * dst);
व्योम ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_get_rows_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_multi_add(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_out_prod(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
पूर्णांक ggml_vk_unary_mul_op_सूचकांक(ggml_unary_op op);
व्योम ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_add_id(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, ggml_tensor * dst);
व्योम ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_gated_linear_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_ssm_scan(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_opt_step_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, ggml_tensor * dst);
व्योम ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_repeat_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_set_rows(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_silu_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_आकार(ggml_backend_vk_context * ctx, स्थिर ggml_tensor *गाँठ);
व्योम ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, भग्न * op_params);
व्योम ggml_vk_rms_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * src2, ggml_tensor * dst);
व्योम ggml_vk_soft_max_back(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, bool backprop);
व्योम ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_योग(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_योग_rows(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_mean(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_cross_entropy_loss_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
व्योम ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_गिनती_equal(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);
व्योम ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst);
व्योम ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, स्थिर ggml_tensor * src0, ggml_tensor * dst);

// graph
व्योम ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, ggml_tensor *गाँठ_begin, पूर्णांक गाँठ_idx_begin, bool last_गाँठ, bool almost_ready, bool submit);
व्योम ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, पूर्णांक tensor_idx, bool almost_ready);
व्योम ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx);
व्योम ggml_vk_cleanup(ggml_backend_vk_context * ctx);
व्योम ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_empty(ggml_tensor * गाँठ);
bool ggml_vk_can_fuse(स्थिर ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, std::initializer_list<क्रमागत ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(स्थिर ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, पूर्णांक अंक_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, topk_moe_mode mode);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
bool ggml_vk_can_fuse_rms_norm_set_rows(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
bool ggml_vk_tensors_overlap(स्थिर ggml_tensor * a, स्थिर ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx);
व्योम ggml_vk_graph_optimize(ggml_backend_t backend, काष्ठा ggml_cgraph * graph, काष्ठा ggml_backend_graph_optimize_params * params);

// backend
bool ggml_backend_बफर_is_vk(ggml_backend_बफर_t बफर);
व्योम ggml_backend_vk_बफर_free_बफर(ggml_backend_बफर_t बफर);
व्योम * ggml_backend_vk_बफर_get_base(ggml_backend_बफर_t बफर);
क्रमागत ggml_status ggml_backend_vk_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor);
व्योम ggml_backend_vk_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t मूल्य, माप_प्रकार offset, माप_प्रकार आकार);
व्योम ggml_backend_vk_बफर_set_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
व्योम ggml_backend_vk_बफर_set_tensor_2d(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य);
व्योम ggml_backend_vk_बफर_get_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
व्योम ggml_backend_vk_बफर_get_tensor_2d(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य);
bool ggml_backend_vk_बफर_cpy_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * src, ggml_tensor * dst);
व्योम ggml_backend_vk_बफर_clear(ggml_backend_बफर_t बफर, uint8_t मूल्य);
स्थिर अक्षर * ggml_backend_vk_बफर_type_name(ggml_backend_बफर_type_t buft);
ggml_backend_बफर_t ggml_backend_vk_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार आकार);
माप_प्रकार ggml_backend_vk_बफर_type_get_alignment(ggml_backend_बफर_type_t buft);
माप_प्रकार ggml_backend_vk_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft);
माप_प्रकार ggml_backend_vk_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor);
व्योम ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_आकार(स्थिर ggml_tensor * op);
#अगर_घोषित GGML_VULKAN_RUN_TESTS
ढाँचा <typename X_TYPE, typename Y_TYPE>
व्योम ggml_vk_test_matmul(ggml_backend_vk_context * ctx, माप_प्रकार m, माप_प्रकार n, माप_प्रकार k, माप_प्रकार batch, माप_प्रकार अंक_it, पूर्णांक split_k, पूर्णांक shader_आकार);
व्योम ggml_vk_test_dequant(ggml_backend_vk_context * ctx, माप_प्रकार ne, ggml_type quant);
व्योम ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, माप_प्रकार m, माप_प्रकार n, माप_प्रकार k, माप_प्रकार batch, माप_प्रकार अंक_it, माप_प्रकार split_k, माप_प्रकार shader_आकार, ggml_type quant, bool mmq = false);
#पूर्ण_अगर
#अगर_घोषित GGML_VULKAN_CHECK_RESULTS
बाह्य माप_प्रकार vk_skip_checks;
बाह्य माप_प्रकार vk_output_tensor;
व्योम ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णांक tensor_idx);
व्योम ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णांक tensor_idx);
#पूर्ण_अगर

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
व्योम ggml_vk_print_device_fault_info(स्थिर vk_device& device);
uint64_t ggml_vk_get_गाँठ_flops(स्थिर ggml_tensor * गाँठ);
व्योम ggml_vk_print_गाँठ_list(स्थिर ggml_cgraph * cgraph, पूर्णांक start, पूर्णांक end);
व्योम ggml_vk_print_device_lost_info(स्थिर vk_device& device);
माप_प्रकार ggml_vk_tensor_बफर_offset(स्थिर ggml_backend_vk_context * ctx, स्थिर ggml_tensor * t);
माप_प्रकार ggml_vk_descriptor_offset(माप_प्रकार tensor_offset, माप_प्रकार alignment, माप_प्रकार type_आकार);
uint32_t ggml_vk_concat_unit_आकार(ggml_type type);
bool ggml_vk_concat_supported(स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, स्थिर ggml_tensor * dst);

ढाँचा <typename T>
अंतरभूत व्योम ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_बफर_Info> स्थिर& descriptor_बफर_infos, स्थिर T &धकेलो_constants, std::array<uint32_t, 3> elements) {
    स्थिर uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    स्थिर uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    स्थिर uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    क्रम (स्वतः& बफर : descriptor_बफर_infos) {
        std::cerr << "(" << बफर.बफर << ", " << बफर.offset << ", " << बफर.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_Group_गिनती[2]);
    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.आकार());
    GGML_ASSERT(descriptor_बफर_infos.आकार() <= MAX_PARAMETER_गिनती);
    GGML_ASSERT(pipeline->parameter_गिनती == descriptor_बफर_infos.आकार());
    GGML_ASSERT(pipeline->धकेलो_constant_आकार == धकेलो_constant_आकार(धकेलो_constants));

    स्थिर uint32_t descriptor_set_idx = ctx->descriptor_set_idx++;
    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[descriptor_set_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    स्थिर uint64_t destroy_गिनती = ctx->device->बफर_destroy_गिनती.load(std::memory_order_acquire);
    अगर (ctx->descriptor_set_bindings_destroy_गिनती != destroy_गिनती) {
        क्रम (स्वतः & b : ctx->descriptor_set_bindings) {
            b.clear();
        }
        ctx->descriptor_set_bindings_destroy_गिनती = destroy_गिनती;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::vector<vk::Descriptor_बफर_Info> & bindings = ctx->descriptor_set_bindings[descriptor_set_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.आकार() == descriptor_बफर_infos.आकार();
    अगर (same) {
        माप_प्रकार i = 0;
        क्रम (स्थिर vk::Descriptor_बफर_Info & info : descriptor_बफर_infos) {
            स्थिर vk::Descriptor_बफर_Info & पूर्व = bindings[i++];
            अगर (पूर्व.बफर != info.बफर || पूर्व.offset != info.offset || पूर्व.range != info.range) {
                same = false;
                अवरोध;
            }
        }
    }
    अगर (!same) {
        vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_गिनती, vk::DescriptorType::e_Storage_बफर, nullptr, descriptor_बफर_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});
        bindings.assign(descriptor_बफर_infos.begin(), descriptor_बफर_infos.end());
    }

    subctx->s->बफर->buf.धकेलो_Constants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, धकेलो_constant_आकार(धकेलो_constants), धकेलो_constant_तथ्य(धकेलो_constants));
    subctx->s->बफर->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->बफर->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_set },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->बफर->buf.dispatch(wg0, wg1, wg2);
    }
}

