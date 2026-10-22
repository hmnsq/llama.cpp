#pragma once
#അന്തര്ഭാവ "ggml-vulkan-push-constants.h"

വരിയിൽ std::ostream & operator<<(std::ostream & os, vk::Buffer buffer) {
    തിരികെ os << static_cast<VkBuffer>(buffer);
}

// shared globals
ബാഹ്യ ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface;
ബാഹ്യ bool vk_memory_logger_enabled;
ബാഹ്യ bool vk_perf_logger_enabled;
ബാഹ്യ bool vk_perf_logger_concurrent;
ബാഹ്യ bool vk_enable_sync_logger;
ബാഹ്യ uint32_t vk_perf_logger_frequency;
ബാഹ്യ std::string vk_pipeline_stats_filter;
ബാഹ്യ ശൂന്യം * സ്ഥിരം vk_ptr_base;
ബാഹ്യ vk_instance_t vk_instance;
ബാഹ്യ ggml_backend_buffer_i ggml_backend_vk_buffer_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
ശൂന്യം ggml_vk_instance_init();
ശൂന്യം ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
പൂർണ്ണസംഖ്യ ggml_vk_get_device_എണ്ണം();
ശൂന്യം ggml_vk_get_device_description(പൂർണ്ണസംഖ്യ device, അക്ഷരം * description, size_t description_size);
bool ggml_vk_instance_layer_settings_available();
bool ggml_vk_instance_portability_enumeration_ext_available(സ്ഥിരം std::vector<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_available(സ്ഥിരം std::vector<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(സ്ഥിരം vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_matrix_support(സ്ഥിരം vk::PhysicalDeviceProperties& props, സ്ഥിരം vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_എണ്ണം(സ്ഥിരം vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_in_range(uint32_t driver_version, uint32_t lower_major, uint32_t lower_minor, uint32_t upper_major, uint32_t upper_minor);

// shaders
ശൂന്യം ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(സ്ഥിരം vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(സ്ഥിരം vk_device& device, സ്ഥിരം vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_mask, bool use_mask_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_size(സ്ഥിരം std::string &pipeline_name, സ്ഥിരം vk_device_architecture &arch);
ശൂന്യം ggml_vk_load_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(സ്ഥിരം vk_device& device, സ്ഥിരം vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(സ്ഥിരം vk_device& device, സ്ഥിരം vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0));
vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size);
ശൂന്യം ggml_vk_destroy_buffer(vk_buffer& buf);
ശൂന്യം * ggml_vk_host_malloc(size_t size);
ശൂന്യം ggml_vk_host_free(ശൂന്യം * ptr);
ശൂന്യം ggml_vk_host_get(സ്ഥിരം vk_device& device, സ്ഥിരം ശൂന്യം * ptr, vk_buffer& buf, size_t& buf_offset);
ശൂന്യം ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size);
ശൂന്യം ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size);
bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false);
bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t size, bool sync_staging = false);
ശൂന്യം ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t spitch, size_t dpitch, size_t width, size_t height);
ശൂന്യം ggml_vk_buffer_write(vk_buffer& dst, size_t offset, സ്ഥിരം ശൂന്യം * src, size_t size);
bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, ശൂന്യം * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false);
ശൂന്യം ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, ശൂന്യം * dst, size_t spitch, size_t dpitch, size_t width, size_t height);
ശൂന്യം ggml_vk_buffer_read(vk_buffer& src, size_t offset, ശൂന്യം * dst, size_t size);
ശൂന്യം ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size);
ശൂന്യം ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size);
ശൂന്യം ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size);
ശൂന്യം ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size);
vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, ശൂന്യം * ptr, size_t size);

// pipelines
uint64_t vk_tensor_offset(സ്ഥിരം ggml_tensor * tensor);
uint32_t get_misalign_bytes(സ്ഥിരം ggml_backend_vk_context * ctx, സ്ഥിരം ggml_tensor * t);
ശൂന്യം ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
ശൂന്യം ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
ശൂന്യം ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
ശൂന്യം ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, സ്ഥിരം vk::QueueFlags& required, സ്ഥിരം vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_അക്കം_queues);
std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, സ്ഥിരം std::unique_ptr<vk_queue>& source);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
ശൂന്യം ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
ശൂന്യം ggml_vk_queue_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(സ്ഥിരം ggml_backend_vk_context* ctx, സ്ഥിരം vk_buffer& buf, size_t offset = 0);
ശൂന്യം ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
ശൂന്യം ggml_vk_set_event(vk_context& ctx, vk::Event& event);
ശൂന്യം ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(സ്ഥിരം ggml_backend_vk_context * ctx, സ്ഥിരം ggml_tensor * tensor, bool allow_misalign = false);
ശൂന്യം ggml_vk_cmd_label_begin(vk::CommandBuffer buf, സ്ഥിരം അക്ഷരം * name);
ശൂന്യം ggml_vk_ctx_end(vk_context& ctx);
ശൂന്യം ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_size(size_t width, size_t align);
ശൂന്യം deferred_memcpy(ശൂന്യം * dst, സ്ഥിരം ശൂന്യം * src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr);
ശൂന്യം deferred_memset(ശൂന്യം * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type);
ശൂന്യം ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_buffer, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_col_bytes, uint64_t d_col_bytes);
bool ggml_vk_dim01_contiguous(സ്ഥിരം ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, സ്ഥിരം ggml_tensor * src, സ്ഥിരം ggml_tensor * dst, ggml_type to);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
ശൂന്യം ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം vk_subbuffer & in, സ്ഥിരം vk_subbuffer & out, uint32_t ne);
ശൂന്യം ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * mixes, സ്ഥിരം ggml_tensor * scale, സ്ഥിരം ggml_tensor * base, ggml_tensor * dst);
ശൂന്യം ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * x, സ്ഥിരം ggml_tensor * weights, ggml_tensor * dst);
ശൂന്യം ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * x, സ്ഥിരം ggml_tensor * residual, സ്ഥിരം ggml_tensor * post, സ്ഥിരം ggml_tensor * comb, ggml_tensor * dst, സ്ഥിരം ggml_tensor * gate_scale_in = nullptr);
ശൂന്യം ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
bool ggml_vk_use_mul_mat_vec_id(സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);

// flash-attn
ശൂന്യം ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * q, സ്ഥിരം ggml_tensor * k, സ്ഥിരം ggml_tensor * v, സ്ഥിരം ggml_tensor * mask, സ്ഥിരം ggml_tensor * sinks, ggml_tensor * dst);

// operators
ശൂന്യം ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം vk_subbuffer & in, സ്ഥിരം vk_subbuffer & out);
bool ggml_vk_can_use_fwht(സ്ഥിരം ggml_backend_vk_context * ctx, സ്ഥിരം ggml_tensor * src1, സ്ഥിരം ggml_tensor * dst);
ശൂന്യം ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst);
ശൂന്യം ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_get_rows_back(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_multi_add(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_out_prod(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
പൂർണ്ണസംഖ്യ ggml_vk_unary_mul_op_index(ggml_unary_op op);
ശൂന്യം ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_add_id(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, സ്ഥിരം ggml_tensor * src2, ggml_tensor * dst);
ശൂന്യം ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_gated_linear_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_ssm_scan(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_opt_step_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, സ്ഥിരം ggml_tensor * src2, ggml_tensor * dst);
ശൂന്യം ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_repeat_back(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_set_rows(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_silu_back(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_size(ggml_backend_vk_context * ctx, സ്ഥിരം ggml_tensor *കണ്ണി);
ശൂന്യം ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, ദശാംശം * op_params);
ശൂന്യം ggml_vk_rms_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, സ്ഥിരം ggml_tensor * src2, ggml_tensor * dst);
ശൂന്യം ggml_vk_soft_max_back(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, bool backprop);
ശൂന്യം ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_sum(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_sum_rows(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_mean(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_cross_entropy_loss_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
ശൂന്യം ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_എണ്ണം_equal(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);
ശൂന്യം ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst);
ശൂന്യം ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, സ്ഥിരം ggml_tensor * src0, ggml_tensor * dst);

// graph
ശൂന്യം ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, ggml_tensor *കണ്ണി_begin, പൂർണ്ണസംഖ്യ കണ്ണി_idx_begin, bool last_കണ്ണി, bool almost_ready, bool submit);
ശൂന്യം ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, പൂർണ്ണസംഖ്യ tensor_idx, bool almost_ready);
ശൂന്യം ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx);
ശൂന്യം ggml_vk_cleanup(ggml_backend_vk_context * ctx);
ശൂന്യം ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_empty(ggml_tensor * കണ്ണി);
bool ggml_vk_can_fuse(സ്ഥിരം ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, std::initializer_list<എണ്ണൽ ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(സ്ഥിരം ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, പൂർണ്ണസംഖ്യ അക്കം_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, topk_moe_mode mode);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
bool ggml_vk_can_fuse_rms_norm_set_rows(ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
bool ggml_vk_tensors_overlap(സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx);
ശൂന്യം ggml_vk_graph_optimize(ggml_backend_t backend, ഘടന ggml_cgraph * graph, ഘടന ggml_backend_graph_optimize_params * params);

// backend
bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t buffer);
ശൂന്യം ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer);
ശൂന്യം * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer);
എണ്ണൽ ggml_status ggml_backend_vk_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor);
ശൂന്യം ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t മൂല്യം, size_t offset, size_t size);
ശൂന്യം ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size);
ശൂന്യം ggml_backend_vk_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
ശൂന്യം ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size);
ശൂന്യം ggml_backend_vk_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
bool ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst);
ശൂന്യം ggml_backend_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t മൂല്യം);
സ്ഥിരം അക്ഷരം * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor);
ശൂന്യം ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_size(സ്ഥിരം ggml_tensor * op);
#ifdef GGML_VULKAN_RUN_TESTS
template <typename X_TYPE, typename Y_TYPE>
ശൂന്യം ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t അക്കം_it, പൂർണ്ണസംഖ്യ split_k, പൂർണ്ണസംഖ്യ shader_size);
ശൂന്യം ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
ശൂന്യം ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t അക്കം_it, size_t split_k, size_t shader_size, ggml_type quant, bool mmq = false);
#endif
#ifdef GGML_VULKAN_CHECK_RESULTS
ബാഹ്യ size_t vk_skip_checks;
ബാഹ്യ size_t vk_output_tensor;
ശൂന്യം ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ tensor_idx);
ശൂന്യം ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ tensor_idx);
#endif

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
ശൂന്യം ggml_vk_print_device_fault_info(സ്ഥിരം vk_device& device);
uint64_t ggml_vk_get_കണ്ണി_flops(സ്ഥിരം ggml_tensor * കണ്ണി);
ശൂന്യം ggml_vk_print_കണ്ണി_list(സ്ഥിരം ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ start, പൂർണ്ണസംഖ്യ end);
ശൂന്യം ggml_vk_print_device_lost_info(സ്ഥിരം vk_device& device);
size_t ggml_vk_tensor_buffer_offset(സ്ഥിരം ggml_backend_vk_context * ctx, സ്ഥിരം ggml_tensor * t);
size_t ggml_vk_descriptor_offset(size_t tensor_offset, size_t alignment, size_t type_size);
uint32_t ggml_vk_concat_unit_size(ggml_type type);
bool ggml_vk_concat_supported(സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, സ്ഥിരം ggml_tensor * dst);

template <typename T>
വരിയിൽ ശൂന്യം ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::DescriptorBufferInfo> സ്ഥിരം& descriptor_buffer_infos, സ്ഥിരം T &തള്ളുക_constants, std::array<uint32_t, 3> elements) {
    സ്ഥിരം uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    സ്ഥിരം uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    സ്ഥിരം uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    വേണ്ടി (auto& buffer : descriptor_buffer_infos) {
        std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_Group_എണ്ണം[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_Group_എണ്ണം[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_Group_എണ്ണം[2]);
    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
    GGML_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_എണ്ണം);
    GGML_ASSERT(pipeline->parameter_എണ്ണം == descriptor_buffer_infos.size());
    GGML_ASSERT(pipeline->തള്ളുക_constant_size == തള്ളുക_constant_size(തള്ളുക_constants));

    സ്ഥിരം uint32_t descriptor_set_idx = ctx->descriptor_set_idx++;
    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[descriptor_set_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    സ്ഥിരം uint64_t destroy_എണ്ണം = ctx->device->buffer_destroy_എണ്ണം.load(std::memory_order_acquire);
    എങ്കിൽ (ctx->descriptor_set_bindings_destroy_എണ്ണം != destroy_എണ്ണം) {
        വേണ്ടി (auto & b : ctx->descriptor_set_bindings) {
            b.clear();
        }
        ctx->descriptor_set_bindings_destroy_എണ്ണം = destroy_എണ്ണം;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::vector<vk::DescriptorBufferInfo> & bindings = ctx->descriptor_set_bindings[descriptor_set_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.size() == descriptor_buffer_infos.size();
    എങ്കിൽ (same) {
        size_t i = 0;
        വേണ്ടി (സ്ഥിരം vk::DescriptorBufferInfo & info : descriptor_buffer_infos) {
            സ്ഥിരം vk::DescriptorBufferInfo & prev = bindings[i++];
            എങ്കിൽ (prev.buffer != info.buffer || prev.offset != info.offset || prev.range != info.range) {
                same = false;
                നിർത്തുക;
            }
        }
    }
    എങ്കിൽ (!same) {
        vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_എണ്ണം, vk::DescriptorType::eStorageBuffer, nullptr, descriptor_buffer_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});
        bindings.assign(descriptor_buffer_infos.begin(), descriptor_buffer_infos.end());
    }

    subctx->s->buffer->buf.തള്ളുക_Constants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, തള്ളുക_constant_size(തള്ളുക_constants), തള്ളുക_constant_data(തള്ളുക_constants));
    subctx->s->buffer->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_set },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->buffer->buf.dispatch(wg0, wg1, wg2);
    }
}

