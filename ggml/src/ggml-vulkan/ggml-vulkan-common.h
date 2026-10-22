#pragma once
#অন্তর্ভাৱ "ggml-vulkan-push-constants.h"

অন্তঃস্থ std::ostream & operator<<(std::ostream & os, vk::Buffer buffer) {
    ঘূৰাই os << static_cast<VkBuffer>(buffer);
}

// shared globals
বাহ্য ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface;
বাহ্য bool vk_memory_logger_enabled;
বাহ্য bool vk_perf_logger_enabled;
বাহ্য bool vk_perf_logger_concurrent;
বাহ্য bool vk_enable_sync_logger;
বাহ্য uint32_t vk_perf_logger_frequency;
বাহ্য std::string vk_pipeline_stats_filter;
বাহ্য শূন্য * ধ্রুৱ vk_ptr_base;
বাহ্য vk_instance_t vk_instance;
বাহ্য ggml_backend_buffer_i ggml_backend_vk_buffer_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
শূন্য ggml_vk_instance_init();
শূন্য ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
পূর্ণাঙ্ক ggml_vk_get_device_গুনতি();
শূন্য ggml_vk_get_device_description(পূর্ণাঙ্ক device, ৱর্ণ * description, size_t description_size);
bool ggml_vk_instance_layer_settings_available();
bool ggml_vk_instance_portability_enumeration_ext_available(ধ্রুৱ std::vector<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_available(ধ্রুৱ std::vector<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(ধ্রুৱ vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_matrix_support(ধ্রুৱ vk::PhysicalDeviceProperties& props, ধ্রুৱ vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_গুনতি(ধ্রুৱ vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_in_range(uint32_t driver_version, uint32_t lower_major, uint32_t lower_minor, uint32_t upper_major, uint32_t upper_minor);

// shaders
শূন্য ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(ধ্রুৱ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(ধ্রুৱ vk_device& device, ধ্রুৱ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_mask, bool use_mask_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_size(ধ্রুৱ std::string &pipeline_name, ধ্রুৱ vk_device_architecture &arch);
শূন্য ggml_vk_load_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(ধ্রুৱ vk_device& device, ধ্রুৱ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(ধ্রুৱ vk_device& device, ধ্রুৱ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0));
vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size);
শূন্য ggml_vk_destroy_buffer(vk_buffer& buf);
শূন্য * ggml_vk_host_malloc(size_t size);
শূন্য ggml_vk_host_free(শূন্য * ptr);
শূন্য ggml_vk_host_get(ধ্রুৱ vk_device& device, ধ্রুৱ শূন্য * ptr, vk_buffer& buf, size_t& buf_offset);
শূন্য ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size);
শূন্য ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size);
bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, ধ্রুৱ শূন্য * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false);
bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, ধ্রুৱ শূন্য * src, size_t size, bool sync_staging = false);
শূন্য ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, ধ্রুৱ শূন্য * src, size_t spitch, size_t dpitch, size_t width, size_t height);
শূন্য ggml_vk_buffer_write(vk_buffer& dst, size_t offset, ধ্রুৱ শূন্য * src, size_t size);
bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, শূন্য * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false);
শূন্য ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, শূন্য * dst, size_t spitch, size_t dpitch, size_t width, size_t height);
শূন্য ggml_vk_buffer_read(vk_buffer& src, size_t offset, শূন্য * dst, size_t size);
শূন্য ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size);
শূন্য ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size);
শূন্য ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size);
শূন্য ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size);
vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, শূন্য * ptr, size_t size);

// pipelines
uint64_t vk_tensor_offset(ধ্রুৱ ggml_tensor * tensor);
uint32_t get_misalign_bytes(ধ্রুৱ ggml_backend_vk_context * ctx, ধ্রুৱ ggml_tensor * t);
শূন্য ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
শূন্য ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
শূন্য ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
শূন্য ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, ধ্রুৱ vk::QueueFlags& required, ধ্রুৱ vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_অঙ্ক_queues);
std::unique_ptr<vk_queue> ggml_vk_create_queue(vk_device& device, uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_queue> ggml_vk_create_aliased_queue(vk_device& device, ধ্রুৱ std::unique_ptr<vk_queue>& source);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
শূন্য ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
শূন্য ggml_vk_queue_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(ধ্রুৱ ggml_backend_vk_context* ctx, ধ্রুৱ vk_buffer& buf, size_t offset = 0);
শূন্য ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
শূন্য ggml_vk_set_event(vk_context& ctx, vk::Event& event);
শূন্য ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(ধ্রুৱ ggml_backend_vk_context * ctx, ধ্রুৱ ggml_tensor * tensor, bool allow_misalign = false);
শূন্য ggml_vk_cmd_label_begin(vk::CommandBuffer buf, ধ্রুৱ ৱর্ণ * name);
শূন্য ggml_vk_ctx_end(vk_context& ctx);
শূন্য ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_size(size_t width, size_t align);
শূন্য deferred_memcpy(শূন্য * dst, ধ্রুৱ শূন্য * src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr);
শূন্য deferred_memset(শূন্য * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type);
শূন্য ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_buffer, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_col_bytes, uint64_t d_col_bytes);
bool ggml_vk_dim01_contiguous(ধ্রুৱ ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ধ্রুৱ ggml_tensor * src, ধ্রুৱ ggml_tensor * dst, ggml_type to);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
শূন্য ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ vk_subbuffer & in, ধ্রুৱ vk_subbuffer & out, uint32_t ne);
শূন্য ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * mixes, ধ্রুৱ ggml_tensor * scale, ধ্রুৱ ggml_tensor * base, ggml_tensor * dst);
শূন্য ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * x, ধ্রুৱ ggml_tensor * weights, ggml_tensor * dst);
শূন্য ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * x, ধ্রুৱ ggml_tensor * residual, ধ্রুৱ ggml_tensor * post, ধ্রুৱ ggml_tensor * comb, ggml_tensor * dst, ধ্রুৱ ggml_tensor * gate_scale_in = nullptr);
শূন্য ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
bool ggml_vk_use_mul_mat_vec_id(ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);

// flash-attn
শূন্য ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * q, ধ্রুৱ ggml_tensor * k, ধ্রুৱ ggml_tensor * v, ধ্রুৱ ggml_tensor * mask, ধ্রুৱ ggml_tensor * sinks, ggml_tensor * dst);

// operators
শূন্য ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ধ্রুৱ ggml_tensor * tensor, ধ্রুৱ vk_subbuffer & in, ধ্রুৱ vk_subbuffer & out);
bool ggml_vk_can_use_fwht(ধ্রুৱ ggml_backend_vk_context * ctx, ধ্রুৱ ggml_tensor * src1, ধ্রুৱ ggml_tensor * dst);
শূন্য ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src, ggml_tensor * dst);
শূন্য ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_get_rows_back(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_multi_add(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_out_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
পূর্ণাঙ্ক ggml_vk_unary_mul_op_index(ggml_unary_op op);
শূন্য ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_add_id(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ধ্রুৱ ggml_tensor * src2, ggml_tensor * dst);
শূন্য ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_gated_linear_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_ssm_scan(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_opt_step_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ধ্রুৱ ggml_tensor * src2, ggml_tensor * dst);
শূন্য ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_repeat_back(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_set_rows(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_silu_back(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_size(ggml_backend_vk_context * ctx, ধ্রুৱ ggml_tensor *গ্ৰন্থি);
শূন্য ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx, ভিন্ন * op_params);
শূন্য ggml_vk_rms_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ধ্রুৱ ggml_tensor * src2, ggml_tensor * dst);
শূন্য ggml_vk_soft_max_back(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx, bool backprop);
শূন্য ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_sum(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_sum_rows(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_mean(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_cross_entropy_loss_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
শূন্য ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_গুনতি_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);
শূন্য ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst);
শূন্য ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ধ্রুৱ ggml_tensor * src0, ggml_tensor * dst);

// graph
শূন্য ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx, ggml_tensor *গ্ৰন্থি_begin, পূর্ণাঙ্ক গ্ৰন্থি_idx_begin, bool last_গ্ৰন্থি, bool almost_ready, bool submit);
শূন্য ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, পূর্ণাঙ্ক tensor_idx, bool almost_ready);
শূন্য ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx);
শূন্য ggml_vk_cleanup(ggml_backend_vk_context * ctx);
শূন্য ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_empty(ggml_tensor * গ্ৰন্থি);
bool ggml_vk_can_fuse(ধ্রুৱ ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx, std::initializer_list<গণনা ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(ধ্রুৱ ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx, পূর্ণাঙ্ক অঙ্ক_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx, topk_moe_mode mode);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
bool ggml_vk_can_fuse_rms_norm_set_rows(ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
bool ggml_vk_tensors_overlap(ধ্রুৱ ggml_tensor * a, ধ্রুৱ ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, ধ্রুৱ সংরচনা ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্ৰন্থি_idx);
শূন্য ggml_vk_graph_optimize(ggml_backend_t backend, সংরচনা ggml_cgraph * graph, সংরচনা ggml_backend_graph_optimize_params * params);

// backend
bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t buffer);
শূন্য ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer);
শূন্য * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer);
গণনা ggml_status ggml_backend_vk_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor);
শূন্য ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t মান, size_t offset, size_t size);
শূন্য ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size);
শূন্য ggml_backend_vk_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
শূন্য ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, ধ্রুৱ ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size);
শূন্য ggml_backend_vk_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, ধ্রুৱ ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
bool ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ধ্রুৱ ggml_tensor * src, ggml_tensor * dst);
শূন্য ggml_backend_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t মান);
ধ্রুৱ ৱর্ণ * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুৱ ggml_tensor * tensor);
শূন্য ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_size(ধ্রুৱ ggml_tensor * op);
#ifdef GGML_VULKAN_RUN_TESTS
template <typename X_TYPE, typename Y_TYPE>
শূন্য ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t অঙ্ক_it, পূর্ণাঙ্ক split_k, পূর্ণাঙ্ক shader_size);
শূন্য ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
শূন্য ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t অঙ্ক_it, size_t split_k, size_t shader_size, ggml_type quant, bool mmq = false);
#endif
#ifdef GGML_VULKAN_CHECK_RESULTS
বাহ্য size_t vk_skip_checks;
বাহ্য size_t vk_output_tensor;
শূন্য ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক tensor_idx);
শূন্য ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক tensor_idx);
#endif

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
শূন্য ggml_vk_print_device_fault_info(ধ্রুৱ vk_device& device);
uint64_t ggml_vk_get_গ্ৰন্থি_flops(ধ্রুৱ ggml_tensor * গ্ৰন্থি);
শূন্য ggml_vk_print_গ্ৰন্থি_list(ধ্রুৱ ggml_cgraph * cgraph, পূর্ণাঙ্ক start, পূর্ণাঙ্ক end);
শূন্য ggml_vk_print_device_lost_info(ধ্রুৱ vk_device& device);
size_t ggml_vk_tensor_buffer_offset(ধ্রুৱ ggml_backend_vk_context * ctx, ধ্রুৱ ggml_tensor * t);
size_t ggml_vk_descriptor_offset(size_t tensor_offset, size_t alignment, size_t type_size);
uint32_t ggml_vk_concat_unit_size(ggml_type type);
bool ggml_vk_concat_supported(ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ধ্রুৱ ggml_tensor * dst);

template <typename T>
অন্তঃস্থ শূন্য ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::DescriptorBufferInfo> ধ্রুৱ& descriptor_buffer_infos, ধ্রুৱ T &ঠেলো_constants, std::array<uint32_t, 3> elements) {
    ধ্রুৱ uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    ধ্রুৱ uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    ধ্রুৱ uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    বাবে (auto& buffer : descriptor_buffer_infos) {
        std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_Group_গুনতি[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_Group_গুনতি[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_Group_গুনতি[2]);
    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
    GGML_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_গুনতি);
    GGML_ASSERT(pipeline->parameter_গুনতি == descriptor_buffer_infos.size());
    GGML_ASSERT(pipeline->ঠেলো_constant_size == ঠেলো_constant_size(ঠেলো_constants));

    ধ্রুৱ uint32_t descriptor_set_idx = ctx->descriptor_set_idx++;
    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[descriptor_set_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    ধ্রুৱ uint64_t destroy_গুনতি = ctx->device->buffer_destroy_গুনতি.load(std::memory_order_acquire);
    যদি (ctx->descriptor_set_bindings_destroy_গুনতি != destroy_গুনতি) {
        বাবে (auto & b : ctx->descriptor_set_bindings) {
            b.clear();
        }
        ctx->descriptor_set_bindings_destroy_গুনতি = destroy_গুনতি;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::vector<vk::DescriptorBufferInfo> & bindings = ctx->descriptor_set_bindings[descriptor_set_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.size() == descriptor_buffer_infos.size();
    যদি (same) {
        size_t i = 0;
        বাবে (ধ্রুৱ vk::DescriptorBufferInfo & info : descriptor_buffer_infos) {
            ধ্রুৱ vk::DescriptorBufferInfo & prev = bindings[i++];
            যদি (prev.buffer != info.buffer || prev.offset != info.offset || prev.range != info.range) {
                same = false;
                ভঙ্গ;
            }
        }
    }
    যদি (!same) {
        vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_গুনতি, vk::DescriptorType::eStorageBuffer, nullptr, descriptor_buffer_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});
        bindings.assign(descriptor_buffer_infos.begin(), descriptor_buffer_infos.end());
    }

    subctx->s->buffer->buf.ঠেলো_Constants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, ঠেলো_constant_size(ঠেলো_constants), ঠেলো_constant_data(ঠেলো_constants));
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

