#ಸೂಚನೆ once
#ಅನ್ತರ್ಭಾವ "ggml-vulkan-push-constants.h"

ಸಾಲೊಳಗೆ std::ostream & ಕ್ರಿಯಾಕಾರಕ<<(std::ostream & os, vk::ಬಫರ್_2 ಬಫರ್) {
    ಹಿಂತಿರುಗಿ os << static_cast<Vk_ಬಫರ್>(ಬಫರ್);
}

// shared globals
ಬಾಹ್ಯ ggml_backend_ಬಫರ್_type_i ggml_backend_vk_ಬಫರ್_type_interface;
ಬಾಹ್ಯ bool vk_ಸ್ಮರಣೆ_logger_enabled;
ಬಾಹ್ಯ bool vk_perf_logger_enabled;
ಬಾಹ್ಯ bool vk_perf_logger_concurrent;
ಬಾಹ್ಯ bool vk_enable_sync_logger;
ಬಾಹ್ಯ uint32_t vk_perf_logger_frequency;
ಬಾಹ್ಯ std::string vk_pipeline_stats_filter;
ಬಾಹ್ಯ ಶೂನ್ಯ * ಸ್ಥಿರಾಂಕ vk_ptr_ತಳ;
ಬಾಹ್ಯ vk_instance_t vk_instance;
ಬಾಹ್ಯ ggml_backend_ಬಫರ್_i ggml_backend_vk_ಬಫರ್_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
ಶೂನ್ಯ ggml_vk_instance_init();
ಶೂನ್ಯ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
ಪೂರ್ಣಾಂಕ ggml_vk_get_device_ಎಣಿಕೆ();
ಶೂನ್ಯ ggml_vk_get_device_description(ಪೂರ್ಣಾಂಕ device, ಅಕ್ಷರ * description, size_t description_ಗಾತ್ರ);
bool ggml_vk_instance_layer_settings_ಲಭ್ಯ();
bool ggml_vk_instance_portability_enumeration_ext_ಲಭ್ಯ(ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_ಲಭ್ಯ(ಸ್ಥಿರಾಂಕ std::ಸದಿಶ<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ಸ್ಥಿರಾಂಕ vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_ಮಾತೃಕೆ_support(ಸ್ಥಿರಾಂಕ vk::PhysicalDeviceProperties& props, ಸ್ಥಿರಾಂಕ vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_ಎಣಿಕೆ(ಸ್ಥಿರಾಂಕ vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_ಒಳ_range(uint32_t driver_ಆವೃತ್ತಿ, uint32_t ಅಧೋ_major, uint32_t ಅಧೋ_minor, uint32_t ಊರ್ಧ್ವ_major, uint32_t ಊರ್ಧ್ವ_minor);

// shaders
ಶೂನ್ಯ ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ಸ್ಥಿರಾಂಕ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_ಮುಖವಾಡ, bool use_ಮುಖವಾಡ_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ std::string &pipeline_name, ಸ್ಥಿರಾಂಕ vk_device_architecture &arch);
ಶೂನ್ಯ ggml_vk_ತುಂಬು_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_ಬಫರ್ ggml_vk_create_ಬಫರ್_check(vk_device& device, size_t ಗಾತ್ರ_2, vk::ಸ್ಮರಣೆ_Property_Flags req_flags, vk::ಸ್ಮರಣೆ_Property_Flags fallback_flags = vk::ಸ್ಮರಣೆ_Property_Flags(0));
vk_ಬಫರ್ ggml_vk_create_ಬಫರ್_device(vk_device& device, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_vk_destroy_ಬಫರ್(vk_ಬಫರ್& buf);
ಶೂನ್ಯ * ggml_vk_host_malloc(size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_vk_host_free(ಶೂನ್ಯ * ptr);
ಶೂನ್ಯ ggml_vk_host_get(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr, vk_ಬಫರ್& buf, size_t& buf_ಸ್ಥಾನಾಂತರ);
ಶೂನ್ಯ ggml_vk_ensure_sync_staging_ಬಫರ್(vk_device& device, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_vk_ensure_sync_staging_ಬಫರ್(ggml_backend_vk_context * ctx, size_t ಗಾತ್ರ_2);
bool ggml_vk_ಬಫರ್_write_2d_async(vk_context subctx, vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ, bool sync_staging = false);
bool ggml_vk_ಬಫರ್_write_async(vk_context subctx, vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t ಗಾತ್ರ_2, bool sync_staging = false);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_write_2d(vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_write(vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t ಗಾತ್ರ_2);
bool ggml_vk_ಬಫರ್_read_2d_async(vk_context subctx, vk_ಬಫರ್& src, size_t ಸ್ಥಾನಾಂತರ, ಶೂನ್ಯ * dst, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ, bool sync_staging = false);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_read_2d(vk_ಬಫರ್& src, size_t ಸ್ಥಾನಾಂತರ, ಶೂನ್ಯ * dst, size_t spitch, size_t dpitch, size_t ಅಗಲ, size_t ಎತ್ತರ);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_read(vk_ಬಫರ್& src, size_t ಸ್ಥಾನಾಂತರ, ಶೂನ್ಯ * dst, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_copy_async(vk_context& ctx, vk_ಬಫರ್& dst, size_t dst_ಸ್ಥಾನಾಂತರ, vk_ಬಫರ್& src, size_t src_ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_copy(vk_ಬಫರ್& dst, size_t dst_ಸ್ಥಾನಾಂತರ, vk_ಬಫರ್& src, size_t src_ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_memset_async(vk_context& ctx, vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, uint32_t c, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_vk_ಬಫರ್_memset(vk_ಬಫರ್& dst, size_t ಸ್ಥಾನಾಂತರ, uint32_t c, size_t ಗಾತ್ರ_2);
vk_ಬಫರ್ ggml_vk_ಬಫರ್_from_host_ptr(vk_device & device, ಶೂನ್ಯ * ptr, size_t ಗಾತ್ರ_2);

// pipelines
uint64_t vk_tensor_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
uint32_t get_misalign_ಬೈಟುಗಳು(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t);
ಶೂನ್ಯ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
ಶೂನ್ಯ ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
ಶೂನ್ಯ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
ಶೂನ್ಯ ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_ಕಂಡುಹಿಡಿ_ಸರತಿ_family_ಸೂಚ್ಯಂಕ(std::ಸದಿಶ<vk::ಸರತಿ_Family_Properties>& ಸರತಿ_family_props, ಸ್ಥಿರಾಂಕ vk::ಸರತಿ_Flags& required, ಸ್ಥಿರಾಂಕ vk::ಸರತಿ_Flags& avoid, int32_t compute_ಸೂಚ್ಯಂಕ, uint32_t min_ಅಂಕೆ_queues);
std::unique_ptr<vk_ಸರತಿ> ggml_vk_create_ಸರತಿ(vk_device& device, uint32_t ಸರತಿ_family_ಸೂಚ್ಯಂಕ, uint32_t ಸರತಿ_ಸೂಚ್ಯಂಕ, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_ಸರತಿ> ggml_vk_create_aliased_ಸರತಿ(vk_device& device, ಸ್ಥಿರಾಂಕ std::unique_ptr<vk_ಸರತಿ>& ಮೂಲ);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
ಶೂನ್ಯ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
ಶೂನ್ಯ ggml_vk_ಸರತಿ_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context* ctx, ಸ್ಥಿರಾಂಕ vk_ಬಫರ್& buf, size_t ಸ್ಥಾನಾಂತರ = 0);
ಶೂನ್ಯ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
ಶೂನ್ಯ ggml_vk_ಗಣ_event(vk_context& ctx, vk::Event& event);
ಶೂನ್ಯ ggml_vk_wait_events(vk_context& ctx, std::ಸದಿಶ<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, bool allow_misalign = false);
ಶೂನ್ಯ ggml_vk_cmd_label_begin(vk::Command_ಬಫರ್ buf, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);
ಶೂನ್ಯ ggml_vk_ctx_ಕೊನೆ(vk_context& ctx);
ಶೂನ್ಯ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_ಗಾತ್ರ(size_t ಅಗಲ, size_t align);
ಶೂನ್ಯ deferred_memcpy(ಶೂನ್ಯ * dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t ಗಾತ್ರ_2, std::ಸದಿಶ<vk_staging_memcpy>* memcpys = nullptr);
ಶೂನ್ಯ deferred_memset(ಶೂನ್ಯ * dst, uint32_t val, size_t ಗಾತ್ರ_2, std::ಸದಿಶ<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_ಗಮ್ಯ_fp16(ggml_backend_vk_context * ctx, ggml_type type);
ಶೂನ್ಯ ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ವಿಭಜಿಸು_k_ಬಫರ್, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t ವಿಭಜಿಸು_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_ಲಂಬಸಾಲು_ಬೈಟುಗಳು, uint64_t d_ಲಂಬಸಾಲು_ಬೈಟುಗಳು);
bool ggml_vk_dim01_contiguous(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ಸ್ಥಿರಾಂಕ ggml_tensor * dst, ggml_type ಗಮ್ಯ);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
ಶೂನ್ಯ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಒಳ, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಹೊರಹೋಗು, uint32_t ne);
ಶೂನ್ಯ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * mixes, ಸ್ಥಿರಾಂಕ ggml_tensor * scale, ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * x, ಸ್ಥಿರಾಂಕ ggml_tensor * weights, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * x, ಸ್ಥಿರಾಂಕ ggml_tensor * residual, ಸ್ಥಿರಾಂಕ ggml_tensor * post, ಸ್ಥಿರಾಂಕ ggml_tensor * comb, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ ggml_tensor * gate_scale_ಒಳ = nullptr);
ಶೂನ್ಯ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
bool ggml_vk_use_mul_mat_vec_id(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);

// flash-attn
ಶೂನ್ಯ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * q, ಸ್ಥಿರಾಂಕ ggml_tensor * k, ಸ್ಥಿರಾಂಕ ggml_tensor * v, ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ, ಸ್ಥಿರಾಂಕ ggml_tensor * sinks, ggml_tensor * dst);

// operators
ಶೂನ್ಯ ggml_vk_cpy_ಗಮ್ಯ_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಒಳ, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಹೊರಹೋಗು);
bool ggml_vk_can_use_fwht(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_multi_ಕೂಡಿಸು(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_ಕೂಡಿಸು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ಹೊರಹೋಗು_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಪೂರ್ಣಾಂಕ ggml_vk_unary_mul_op_ಸೂಚ್ಯಂಕ(ggml_unary_op op);
ಶೂನ್ಯ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ಕೂಡಿಸು_id(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_gated_ರೇಖೀಯ_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ssm_ಪರಿಶೀಲನೆ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_opt_ಹೆಜ್ಜೆ_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_opt_ಹೆಜ್ಜೆ_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_repeat_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_silu_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ಗುಂಪು_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_ಗಾತ್ರ(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *ಗಂಟು);
ಶೂನ್ಯ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ದಶಮಾಂಶ * op_params);
ಶೂನ್ಯ ggml_vk_rms_norm_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_diag_ಮುಖವಾಡ_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_soft_max_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, bool backprop);
ಶೂನ್ಯ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_ಮೊತ್ತ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ಸರಾಸರಿ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_cross_entropy_loss_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_ಎಣಿಕೆ_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst);
ಶೂನ್ಯ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst);

//ನಕ್ಷೆ
ಶೂನ್ಯ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_ನಕ್ಷೆ(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ggml_tensor *ಗಂಟು_begin, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx_begin, bool ಕೊನೆಯ_ಗಂಟು, bool almost_ready, bool submit);
ಶೂನ್ಯ ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, ಪೂರ್ಣಾಂಕ tensor_idx, bool almost_ready);
ಶೂನ್ಯ ggml_vk_ನಕ್ಷೆ_cleanup(ggml_backend_vk_context * ctx);
ಶೂನ್ಯ ggml_vk_cleanup(ggml_backend_vk_context * ctx);
ಶೂನ್ಯ ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_ಖಾಲಿಗಳು(ggml_tensor * ಗಂಟು);
bool ggml_vk_can_fuse(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, std::initializer_list<ಎಣಿಕೆ ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, topk_moe_ಬಹುಲಕ ಬಹುಲಕ);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
bool ggml_vk_can_fuse_rope_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
bool ggml_vk_can_fuse_rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
bool ggml_vk_tensors_overlap(ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
uint32_t ggml_vk_fuse_multi_ಕೂಡಿಸು(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
ಶೂನ್ಯ ggml_vk_ನಕ್ಷೆ_optimize(ggml_backend_t backend, ರಚನೆ ggml_cgraph * ನಕ್ಷೆ, ರಚನೆ ggml_backend_ನಕ್ಷೆ_optimize_params * params);

// backend
bool ggml_backend_ಬಫರ್_is_vk(ggml_backend_ಬಫರ್_t ಬಫರ್);
ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್);
ಶೂನ್ಯ * ggml_backend_vk_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್);
ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_vk_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor);
ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t ಮೌಲ್ಯ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_ಗಣ_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ);
ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_get_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ);
bool ggml_backend_vk_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst);
ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t ಮೌಲ್ಯ);
ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_vk_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t buft);
ggml_backend_ಬಫರ್_t ggml_backend_vk_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t ಗಾತ್ರ_2);
size_t ggml_backend_vk_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft);
size_t ggml_backend_vk_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft);
size_t ggml_backend_vk_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
ಶೂನ್ಯ ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ggml_tensor * op);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_ಸತತ_TESTS
ಮಾದರಿ <typename X_TYPE, typename Y_TYPE>
ಶೂನ್ಯ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_k, ಪೂರ್ಣಾಂಕ shader_ಗಾತ್ರ);
ಶೂನ್ಯ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
ಶೂನ್ಯ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, size_t ವಿಭಜಿಸು_k, size_t shader_ಗಾತ್ರ, ggml_type quant, bool mmq = false);
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_CHECK_RESULTS
ಬಾಹ್ಯ size_t vk_skip_checks;
ಬಾಹ್ಯ size_t vk_output_tensor;
ಶೂನ್ಯ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ tensor_idx);
ಶೂನ್ಯ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ tensor_idx);
#ಆದರೆ_ಅಂತ್ಯ

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
ಶೂನ್ಯ ggml_vk_print_device_fault_info(ಸ್ಥಿರಾಂಕ vk_device& device);
uint64_t ggml_vk_get_ಗಂಟು_flops(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು);
ಶೂನ್ಯ ggml_vk_print_ಗಂಟು_list(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಆರಂಭ, ಪೂರ್ಣಾಂಕ ಕೊನೆ);
ಶೂನ್ಯ ggml_vk_print_device_ಕಳೆದುಹೋದ_info(ಸ್ಥಿರಾಂಕ vk_device& device);
size_t ggml_vk_tensor_ಬಫರ್_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t);
size_t ggml_vk_descriptor_ಸ್ಥಾನಾಂತರ(size_t tensor_ಸ್ಥಾನಾಂತರ, size_t alignment, size_t type_ಗಾತ್ರ);
uint32_t ggml_vk_concat_unit_ಗಾತ್ರ(ggml_type type);
bool ggml_vk_concat_supported(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * dst);

ಮಾದರಿ <typename T>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_ಬಫರ್_Info> ಸ್ಥಿರಾಂಕ& descriptor_ಬಫರ್_infos, ಸ್ಥಿರಾಂಕ T &ತಳ್ಳು_constants, std::array<uint32_t, 3> elements) {
    ಸ್ಥಿರಾಂಕ uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ಸ್ಥಿರಾಂಕ uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ಸ್ಥಿರಾಂಕ uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& ಬಫರ್ : descriptor_ಬಫರ್_infos) {
        std::cerr << "(" << ಬಫರ್.ಬಫರ್ << ", " << ಬಫರ್.ಸ್ಥಾನಾಂತರ << ", " << ಬಫರ್.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);
    GGML_ASSERT(ctx->descriptor_ಗಣ_idx < ctx->descriptor_sets.ಗಾತ್ರ_2());
    GGML_ASSERT(descriptor_ಬಫರ್_infos.ಗಾತ್ರ_2() <= MAX_PARAMETER_ಎಣಿಕೆ);
    GGML_ASSERT(pipeline->parameter_ಎಣಿಕೆ == descriptor_ಬಫರ್_infos.ಗಾತ್ರ_2());
    GGML_ASSERT(pipeline->ತಳ್ಳು_constant_ಗಾತ್ರ == ತಳ್ಳು_constant_ಗಾತ್ರ(ತಳ್ಳು_constants));

    ಸ್ಥಿರಾಂಕ uint32_t descriptor_ಗಣ_idx = ctx->descriptor_ಗಣ_idx++;
    vk::Descriptor_ಗಣ& descriptor_ಗಣ = ctx->descriptor_sets[descriptor_ಗಣ_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ಸ್ಥಿರಾಂಕ uint64_t destroy_ಎಣಿಕೆ = ctx->device->ಬಫರ್_destroy_ಎಣಿಕೆ.ತುಂಬು(std::memory_order_acquire);
    ಆದರೆ (ctx->descriptor_ಗಣ_bindings_destroy_ಎಣಿಕೆ != destroy_ಎಣಿಕೆ) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & b : ctx->descriptor_ಗಣ_bindings) {
            b.clear();
        }
        ctx->descriptor_ಗಣ_bindings_destroy_ಎಣಿಕೆ = destroy_ಎಣಿಕೆ;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::ಸದಿಶ<vk::Descriptor_ಬಫರ್_Info> & bindings = ctx->descriptor_ಗಣ_bindings[descriptor_ಗಣ_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.ಗಾತ್ರ_2() == descriptor_ಬಫರ್_infos.ಗಾತ್ರ_2();
    ಆದರೆ (same) {
        size_t i = 0;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ vk::Descriptor_ಬಫರ್_Info & info : descriptor_ಬಫರ್_infos) {
            ಸ್ಥಿರಾಂಕ vk::Descriptor_ಬಫರ್_Info & prev = bindings[i++];
            ಆದರೆ (prev.ಬಫರ್ != info.ಬಫರ್ || prev.ಸ್ಥಾನಾಂತರ != info.ಸ್ಥಾನಾಂತರ || prev.range != info.range) {
                same = false;
                ನಿಲ್ಲಿಸು;
            }
        }
    }
    ಆದರೆ (!same) {
        vk::Write_Descriptor_ಗಣ write_descriptor_ಗಣ{ descriptor_ಗಣ, 0, 0, pipeline->parameter_ಎಣಿಕೆ, vk::DescriptorType::e_Storage_ಬಫರ್, nullptr, descriptor_ಬಫರ್_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_ಗಣ }, {});
        bindings.ನಿಯೋಜಿಸು(descriptor_ಬಫರ್_infos.begin(), descriptor_ಬಫರ್_infos.ಕೊನೆ());
    }

    subctx->s->ಬಫರ್->buf.ತಳ್ಳು_Constants(pipeline->layout, vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute, 0, ತಳ್ಳು_constant_ಗಾತ್ರ(ತಳ್ಳು_constants), ತಳ್ಳು_constant_ದತ್ತಾಂಶ(ತಳ್ಳು_constants));
    subctx->s->ಬಫರ್->buf.bindPipeline(vk::Pipeline_Bind_ಬಿಂದು::eCompute, pipeline->pipeline);
    subctx->s->ಬಫರ್->buf.bindDescriptorSets(vk::Pipeline_Bind_ಬಿಂದು::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_ಗಣ },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->ಬಫರ್->buf.dispatch(wg0, wg1, wg2);
    }
}

