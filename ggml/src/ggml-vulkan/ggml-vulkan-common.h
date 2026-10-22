#ਹਦਾਇਤ once
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan-push-constants.h"

ਸਤਰਅੰਦਰ std::ostream & ਸੰਚਾਲਕ<<(std::ostream & os, vk::ਬਫਰ_2 ਬਫਰ) {
    ਵਾਪਸ os << static_cast<Vk_ਬਫਰ>(ਬਫਰ);
}

// shared globals
ਬਾਹਰੀ ggml_backend_ਬਫਰ_type_i ggml_backend_vk_ਬਫਰ_type_interface;
ਬਾਹਰੀ bool vk_ਸਮ੍ਰਿਤੀ_logger_enabled;
ਬਾਹਰੀ bool vk_perf_logger_enabled;
ਬਾਹਰੀ bool vk_perf_logger_concurrent;
ਬਾਹਰੀ bool vk_enable_sync_logger;
ਬਾਹਰੀ uint32_t vk_perf_logger_frequency;
ਬਾਹਰੀ std::string vk_pipeline_stats_filter;
ਬਾਹਰੀ ਖਾਲੀ * ਅਚਲ vk_ptr_ਅਧਾਰ;
ਬਾਹਰੀ vk_instance_t vk_instance;
ਬਾਹਰੀ ggml_backend_ਬਫਰ_i ggml_backend_vk_ਬਫਰ_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
ਖਾਲੀ ggml_vk_instance_init();
ਖਾਲੀ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
ਪੂਰਨ ggml_vk_get_device_ਗਿਣਤੀ();
ਖਾਲੀ ggml_vk_get_device_description(ਪੂਰਨ device, ਅੱਖਰ * description, size_t description_ਮਾਪ);
bool ggml_vk_instance_layer_settings_ਉਪਲਬਧ();
bool ggml_vk_instance_portability_enumeration_ext_ਉਪਲਬਧ(ਅਚਲ std::ਸਦਿਸ਼<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_ਉਪਲਬਧ(ਅਚਲ std::ਸਦਿਸ਼<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ਅਚਲ vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_ਮੈਟ੍ਰਿਕਸ_support(ਅਚਲ vk::PhysicalDeviceProperties& props, ਅਚਲ vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_ਗਿਣਤੀ(ਅਚਲ vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_ਵਿੱਚ_range(uint32_t driver_ਸੰਸਕਰਣ, uint32_t ਹੇਠਲਾ_major, uint32_t ਹੇਠਲਾ_minor, uint32_t ਉੱਤਲਾ_major, uint32_t ਉੱਤਲਾ_minor);

// shaders
ਖਾਲੀ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ਅਚਲ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ਪੰਕਤੀਆਂ, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ਅਚਲ vk_device& device, ਅਚਲ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_ਨਕਾਬ, bool use_ਨਕਾਬ_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_ਮਾਪ(ਅਚਲ std::string &pipeline_name, ਅਚਲ vk_device_architecture &arch);
ਖਾਲੀ ggml_vk_ਲੱਦੋ_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ਅਚਲ vk_device& device, ਅਚਲ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ਅਚਲ vk_device& device, ਅਚਲ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_ਬਫਰ ggml_vk_create_ਬਫਰ_check(vk_device& device, size_t ਮਾਪ, vk::ਸਮ੍ਰਿਤੀ_Property_Flags req_flags, vk::ਸਮ੍ਰਿਤੀ_Property_Flags fallback_flags = vk::ਸਮ੍ਰਿਤੀ_Property_Flags(0));
vk_ਬਫਰ ggml_vk_create_ਬਫਰ_device(vk_device& device, size_t ਮਾਪ);
ਖਾਲੀ ggml_vk_destroy_ਬਫਰ(vk_ਬਫਰ& buf);
ਖਾਲੀ * ggml_vk_host_malloc(size_t ਮਾਪ);
ਖਾਲੀ ggml_vk_host_free(ਖਾਲੀ * ptr);
ਖਾਲੀ ggml_vk_host_get(ਅਚਲ vk_device& device, ਅਚਲ ਖਾਲੀ * ptr, vk_ਬਫਰ& buf, size_t& buf_ਵਿੱਥ);
ਖਾਲੀ ggml_vk_ensure_sync_staging_ਬਫਰ(vk_device& device, size_t ਮਾਪ);
ਖਾਲੀ ggml_vk_ensure_sync_staging_ਬਫਰ(ggml_backend_vk_context * ctx, size_t ਮਾਪ);
bool ggml_vk_ਬਫਰ_write_2d_async(vk_context subctx, vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ, bool sync_staging = false);
bool ggml_vk_ਬਫਰ_write_async(vk_context subctx, vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t ਮਾਪ, bool sync_staging = false);
ਖਾਲੀ ggml_vk_ਬਫਰ_write_2d(vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ);
ਖਾਲੀ ggml_vk_ਬਫਰ_write(vk_ਬਫਰ& dst, size_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * src, size_t ਮਾਪ);
bool ggml_vk_ਬਫਰ_read_2d_async(vk_context subctx, vk_ਬਫਰ& src, size_t ਵਿੱਥ, ਖਾਲੀ * dst, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ, bool sync_staging = false);
ਖਾਲੀ ggml_vk_ਬਫਰ_read_2d(vk_ਬਫਰ& src, size_t ਵਿੱਥ, ਖਾਲੀ * dst, size_t spitch, size_t dpitch, size_t ਚੌੜਾਈ, size_t ਉਚਾਈ);
ਖਾਲੀ ggml_vk_ਬਫਰ_read(vk_ਬਫਰ& src, size_t ਵਿੱਥ, ਖਾਲੀ * dst, size_t ਮਾਪ);
ਖਾਲੀ ggml_vk_ਬਫਰ_copy_async(vk_context& ctx, vk_ਬਫਰ& dst, size_t dst_ਵਿੱਥ, vk_ਬਫਰ& src, size_t src_ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ ggml_vk_ਬਫਰ_copy(vk_ਬਫਰ& dst, size_t dst_ਵਿੱਥ, vk_ਬਫਰ& src, size_t src_ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ ggml_vk_ਬਫਰ_memset_async(vk_context& ctx, vk_ਬਫਰ& dst, size_t ਵਿੱਥ, uint32_t c, size_t ਮਾਪ);
ਖਾਲੀ ggml_vk_ਬਫਰ_memset(vk_ਬਫਰ& dst, size_t ਵਿੱਥ, uint32_t c, size_t ਮਾਪ);
vk_ਬਫਰ ggml_vk_ਬਫਰ_from_host_ptr(vk_device & device, ਖਾਲੀ * ptr, size_t ਮਾਪ);

// pipelines
uint64_t vk_tensor_ਵਿੱਥ(ਅਚਲ ggml_tensor * tensor);
uint32_t get_misalign_ਬਾਈਟਾਂ(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * t);
ਖਾਲੀ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
ਖਾਲੀ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
ਖਾਲੀ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
ਖਾਲੀ ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_ਲੱਭੋ_ਕਤਾਰ_family_ਸੂਚਕਾਂਕ(std::ਸਦਿਸ਼<vk::ਕਤਾਰ_Family_Properties>& ਕਤਾਰ_family_props, ਅਚਲ vk::ਕਤਾਰ_Flags& required, ਅਚਲ vk::ਕਤਾਰ_Flags& avoid, int32_t compute_ਸੂਚਕਾਂਕ, uint32_t min_ਅੰਕ_queues);
std::unique_ptr<vk_ਕਤਾਰ> ggml_vk_create_ਕਤਾਰ(vk_device& device, uint32_t ਕਤਾਰ_family_ਸੂਚਕਾਂਕ, uint32_t ਕਤਾਰ_ਸੂਚਕਾਂਕ, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_ਕਤਾਰ> ggml_vk_create_aliased_ਕਤਾਰ(vk_device& device, ਅਚਲ std::unique_ptr<vk_ਕਤਾਰ>& ਸਰੋਤ);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
ਖਾਲੀ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
ਖਾਲੀ ggml_vk_ਕਤਾਰ_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ਅਚਲ ggml_backend_vk_context* ctx, ਅਚਲ vk_ਬਫਰ& buf, size_t ਵਿੱਥ = 0);
ਖਾਲੀ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
ਖਾਲੀ ggml_vk_ਸੈੱਟ_event(vk_context& ctx, vk::Event& event);
ਖਾਲੀ ggml_vk_wait_events(vk_context& ctx, std::ਸਦਿਸ਼<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * tensor, bool allow_misalign = false);
ਖਾਲੀ ggml_vk_cmd_label_begin(vk::Command_ਬਫਰ buf, ਅਚਲ ਅੱਖਰ * name);
ਖਾਲੀ ggml_vk_ctx_ਸਿਰਾ(vk_context& ctx);
ਖਾਲੀ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_ਮਾਪ(size_t ਚੌੜਾਈ, size_t align);
ਖਾਲੀ deferred_memcpy(ਖਾਲੀ * dst, ਅਚਲ ਖਾਲੀ * src, size_t ਮਾਪ, std::ਸਦਿਸ਼<vk_staging_memcpy>* memcpys = nullptr);
ਖਾਲੀ deferred_memset(ਖਾਲੀ * dst, uint32_t val, size_t ਮਾਪ, std::ਸਦਿਸ਼<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ggml_backend_vk_context * ctx, ggml_type type);
ਖਾਲੀ ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ਵੰਡੋ_k_ਬਫਰ, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t ਵੰਡੋ_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_ਖਾਨਾ_ਬਾਈਟਾਂ, uint64_t d_ਖਾਨਾ_ਬਾਈਟਾਂ);
bool ggml_vk_dim01_contiguous(ਅਚਲ ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * src, ਅਚਲ ggml_tensor * dst, ggml_type ਮੰਜ਼ਿਲ);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
ਖਾਲੀ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ vk_subbuffer & ਵਿੱਚ, ਅਚਲ vk_subbuffer & ਬਾਹਰਲਾ, uint32_t ne);
ਖਾਲੀ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * mixes, ਅਚਲ ggml_tensor * scale, ਅਚਲ ggml_tensor * ਅਧਾਰ, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * x, ਅਚਲ ggml_tensor * weights, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * x, ਅਚਲ ggml_tensor * residual, ਅਚਲ ggml_tensor * post, ਅਚਲ ggml_tensor * comb, ggml_tensor * dst, ਅਚਲ ggml_tensor * gate_scale_ਵਿੱਚ = nullptr);
ਖਾਲੀ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
bool ggml_vk_use_mul_mat_vec_id(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);

// flash-attn
ਖਾਲੀ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * q, ਅਚਲ ggml_tensor * k, ਅਚਲ ggml_tensor * v, ਅਚਲ ggml_tensor * ਨਕਾਬ, ਅਚਲ ggml_tensor * sinks, ggml_tensor * dst);

// operators
ਖਾਲੀ ggml_vk_cpy_ਮੰਜ਼ਿਲ_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ਅਚਲ ggml_tensor * tensor, ਅਚਲ vk_subbuffer & ਵਿੱਚ, ਅਚਲ vk_subbuffer & ਬਾਹਰਲਾ);
bool ggml_vk_can_use_fwht(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * dst);
ਖਾਲੀ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_get_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_multi_ਜੋੜੋ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_ਜੋੜੋ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ਬਾਹਰਲਾ_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਪੂਰਨ ggml_vk_unary_mul_op_ਸੂਚਕਾਂਕ(ggml_unary_op op);
ਖਾਲੀ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ਜੋੜੋ_id(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_gated_ਰੇਖਿਕ_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ssm_ਜਾਂਚ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_opt_ਕਦਮ_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_opt_ਕਦਮ_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_repeat_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_silu_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ਸਮੂਹ_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_ਮਾਪ(ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor *ਗੰਢ);
ਖਾਲੀ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਦਸ਼ਮਲਵ * op_params);
ਖਾਲੀ ggml_vk_rms_norm_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_diag_ਨਕਾਬ_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * src2, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_soft_max_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, bool backprop);
ਖਾਲੀ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_ਜੋੜ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ਔਸਤ(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_cross_entropy_loss_ਪਿੱਠ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_ਗਿਣਤੀ_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst);
ਖਾਲੀ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ਅਚਲ ggml_tensor * src0, ggml_tensor * dst);

//ਗ੍ਰਾਫ
ਖਾਲੀ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_ਗ੍ਰਾਫ(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ggml_tensor *ਗੰਢ_begin, ਪੂਰਨ ਗੰਢ_idx_begin, bool ਆਖਰੀ_ਗੰਢ, bool almost_ready, bool submit);
ਖਾਲੀ ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, ਪੂਰਨ tensor_idx, bool almost_ready);
ਖਾਲੀ ggml_vk_ਗ੍ਰਾਫ_cleanup(ggml_backend_vk_context * ctx);
ਖਾਲੀ ggml_vk_cleanup(ggml_backend_vk_context * ctx);
ਖਾਲੀ ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_ਖਾਲੀ_ਥਾਵਾਂ(ggml_tensor * ਗੰਢ);
bool ggml_vk_can_fuse(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, std::initializer_list<ਗਿਣਤੀ ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਪੂਰਨ ਅੰਕ_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, topk_moe_ਬਹੁਲਕ ਬਹੁਲਕ);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
bool ggml_vk_can_fuse_rope_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
bool ggml_vk_can_fuse_rms_norm_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
bool ggml_vk_tensors_overlap(ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
uint32_t ggml_vk_fuse_multi_ਜੋੜੋ(ggml_backend_vk_context * ctx, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx);
ਖਾਲੀ ggml_vk_ਗ੍ਰਾਫ_optimize(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ, ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_optimize_params * params);

// backend
bool ggml_backend_ਬਫਰ_is_vk(ggml_backend_ਬਫਰ_t ਬਫਰ);
ਖਾਲੀ ggml_backend_vk_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ);
ਖਾਲੀ * ggml_backend_vk_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ);
ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_vk_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor);
ਖਾਲੀ ggml_backend_vk_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t ਮੁੱਲ, size_t ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ ggml_backend_vk_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ ggml_backend_vk_ਬਫਰ_ਸੈੱਟ_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ);
ਖਾਲੀ ggml_backend_vk_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
ਖਾਲੀ ggml_backend_vk_ਬਫਰ_get_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ);
bool ggml_backend_vk_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * src, ggml_tensor * dst);
ਖਾਲੀ ggml_backend_vk_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t ਮੁੱਲ);
ਅਚਲ ਅੱਖਰ * ggml_backend_vk_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t buft);
ggml_backend_ਬਫਰ_t ggml_backend_vk_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t ਮਾਪ);
size_t ggml_backend_vk_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft);
size_t ggml_backend_vk_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft);
size_t ggml_backend_vk_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor);
ਖਾਲੀ ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_ਮਾਪ(ਅਚਲ ggml_tensor * op);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_ਲੜੀਵਾਰ_TESTS
ਸਾਂਚਾ <typename X_TYPE, typename Y_TYPE>
ਖਾਲੀ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ਅੰਕ_it, ਪੂਰਨ ਵੰਡੋ_k, ਪੂਰਨ shader_ਮਾਪ);
ਖਾਲੀ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
ਖਾਲੀ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ਅੰਕ_it, size_t ਵੰਡੋ_k, size_t shader_ਮਾਪ, ggml_type quant, bool mmq = false);
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_CHECK_RESULTS
ਬਾਹਰੀ size_t vk_skip_checks;
ਬਾਹਰੀ size_t vk_output_tensor;
ਖਾਲੀ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ਪੂਰਨ tensor_idx);
ਖਾਲੀ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ਪੂਰਨ tensor_idx);
#ਜੇ_ਅੰਤ

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
ਖਾਲੀ ggml_vk_print_device_fault_info(ਅਚਲ vk_device& device);
uint64_t ggml_vk_get_ਗੰਢ_flops(ਅਚਲ ggml_tensor * ਗੰਢ);
ਖਾਲੀ ggml_vk_print_ਗੰਢ_list(ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਸ਼ੁਰੂ, ਪੂਰਨ ਸਿਰਾ);
ਖਾਲੀ ggml_vk_print_device_ਗੁਆਚਾ_info(ਅਚਲ vk_device& device);
size_t ggml_vk_tensor_ਬਫਰ_ਵਿੱਥ(ਅਚਲ ggml_backend_vk_context * ctx, ਅਚਲ ggml_tensor * t);
size_t ggml_vk_descriptor_ਵਿੱਥ(size_t tensor_ਵਿੱਥ, size_t alignment, size_t type_ਮਾਪ);
uint32_t ggml_vk_concat_unit_ਮਾਪ(ggml_type type);
bool ggml_vk_concat_supported(ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * dst);

ਸਾਂਚਾ <typename T>
ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_ਬਫਰ_Info> ਅਚਲ& descriptor_ਬਫਰ_infos, ਅਚਲ T &ਧੱਕੋ_constants, std::array<uint32_t, 3> elements) {
    ਅਚਲ uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ਅਚਲ uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ਅਚਲ uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    ਲਈ (ਸਵੈਚਾਲਿਤ& ਬਫਰ : descriptor_ਬਫਰ_infos) {
        std::cerr << "(" << ਬਫਰ.ਬਫਰ << ", " << ਬਫਰ.ਵਿੱਥ << ", " << ਬਫਰ.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_ਸਮੂਹ_ਗਿਣਤੀ[2]);
    GGML_ASSERT(ctx->descriptor_ਸੈੱਟ_idx < ctx->descriptor_sets.ਮਾਪ());
    GGML_ASSERT(descriptor_ਬਫਰ_infos.ਮਾਪ() <= MAX_PARAMETER_ਗਿਣਤੀ);
    GGML_ASSERT(pipeline->parameter_ਗਿਣਤੀ == descriptor_ਬਫਰ_infos.ਮਾਪ());
    GGML_ASSERT(pipeline->ਧੱਕੋ_constant_ਮਾਪ == ਧੱਕੋ_constant_ਮਾਪ(ਧੱਕੋ_constants));

    ਅਚਲ uint32_t descriptor_ਸੈੱਟ_idx = ctx->descriptor_ਸੈੱਟ_idx++;
    vk::Descriptor_ਸੈੱਟ& descriptor_ਸੈੱਟ = ctx->descriptor_sets[descriptor_ਸੈੱਟ_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ਅਚਲ uint64_t destroy_ਗਿਣਤੀ = ctx->device->ਬਫਰ_destroy_ਗਿਣਤੀ.ਲੱਦੋ(std::memory_order_acquire);
    ਜੇ (ctx->descriptor_ਸੈੱਟ_bindings_destroy_ਗਿਣਤੀ != destroy_ਗਿਣਤੀ) {
        ਲਈ (ਸਵੈਚਾਲਿਤ & b : ctx->descriptor_ਸੈੱਟ_bindings) {
            b.clear();
        }
        ctx->descriptor_ਸੈੱਟ_bindings_destroy_ਗਿਣਤੀ = destroy_ਗਿਣਤੀ;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::ਸਦਿਸ਼<vk::Descriptor_ਬਫਰ_Info> & bindings = ctx->descriptor_ਸੈੱਟ_bindings[descriptor_ਸੈੱਟ_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.ਮਾਪ() == descriptor_ਬਫਰ_infos.ਮਾਪ();
    ਜੇ (same) {
        size_t i = 0;
        ਲਈ (ਅਚਲ vk::Descriptor_ਬਫਰ_Info & info : descriptor_ਬਫਰ_infos) {
            ਅਚਲ vk::Descriptor_ਬਫਰ_Info & prev = bindings[i++];
            ਜੇ (prev.ਬਫਰ != info.ਬਫਰ || prev.ਵਿੱਥ != info.ਵਿੱਥ || prev.range != info.range) {
                same = false;
                ਰੋਕੋ;
            }
        }
    }
    ਜੇ (!same) {
        vk::Write_Descriptor_ਸੈੱਟ write_descriptor_ਸੈੱਟ{ descriptor_ਸੈੱਟ, 0, 0, pipeline->parameter_ਗਿਣਤੀ, vk::DescriptorType::e_Storage_ਬਫਰ, nullptr, descriptor_ਬਫਰ_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_ਸੈੱਟ }, {});
        bindings.ਸੌਂਪੋ(descriptor_ਬਫਰ_infos.begin(), descriptor_ਬਫਰ_infos.ਸਿਰਾ());
    }

    subctx->s->ਬਫਰ->buf.ਧੱਕੋ_Constants(pipeline->layout, vk::Shader_Stage_Flag_ਬਿੱਟ::eCompute, 0, ਧੱਕੋ_constant_ਮਾਪ(ਧੱਕੋ_constants), ਧੱਕੋ_constant_ਅੰਕੜੇ(ਧੱਕੋ_constants));
    subctx->s->ਬਫਰ->buf.bindPipeline(vk::Pipeline_Bind_ਬਿੰਦੂ::eCompute, pipeline->pipeline);
    subctx->s->ਬਫਰ->buf.bindDescriptorSets(vk::Pipeline_Bind_ਬਿੰਦੂ::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_ਸੈੱਟ },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->ਬਫਰ->buf.dispatch(wg0, wg1, wg2);
    }
}

