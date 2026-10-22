#సూచన once
#అన్తర్భావ "ggml-vulkan-push-constants.h"

అంతఃవరుస std::ostream & క్రియాకారకం<<(std::ostream & os, vk::బఫరు_2 బఫరు) {
    తిరిగి os << static_cast<Vk_బఫరు>(బఫరు);
}

// shared globals
బాహ్య ggml_backend_బఫరు_type_i ggml_backend_vk_బఫరు_type_interface;
బాహ్య bool vk_జ్ఞాపకం_logger_enabled;
బాహ్య bool vk_perf_logger_enabled;
బాహ్య bool vk_perf_logger_concurrent;
బాహ్య bool vk_enable_sync_logger;
బాహ్య uint32_t vk_perf_logger_frequency;
బాహ్య std::string vk_pipeline_stats_filter;
బాహ్య శూన్య * స్థిరరాశి vk_ptr_పునాది;
బాహ్య vk_instance_t vk_instance;
బాహ్య ggml_backend_బఫరు_i ggml_backend_vk_బఫరు_interface;

// instance
vk_device ggml_vk_get_device(size_t idx);
DispatchLoaderDynamic & ggml_vk_default_dispatcher();
శూన్య ggml_vk_instance_init();
శూన్య ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx);
పూర్ణాంకం ggml_vk_get_device_లెక్క();
శూన్య ggml_vk_get_device_description(పూర్ణాంకం device, అక్షరం * description, size_t description_కొలత);
bool ggml_vk_instance_layer_settings_అందుబాటు();
bool ggml_vk_instance_portability_enumeration_ext_అందుబాటు(స్థిరరాశి std::సదిశ<vk::ExtensionProperties>& instance_extensions);
bool ggml_vk_instance_debug_utils_ext_అందుబాటు(స్థిరరాశి std::సదిశ<vk::ExtensionProperties> & instance_extensions);
bool ggml_vk_device_is_supported(స్థిరరాశి vk::PhysicalDevice & vkdev);
bool ggml_vk_khr_cooperative_మాతృక_support(స్థిరరాశి vk::PhysicalDeviceProperties& props, స్థిరరాశి vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);
uint32_t ggml_vk_intel_shader_core_లెక్క(స్థిరరాశి vk::PhysicalDevice& vkdev);
bool ggml_vk_intel_windows_driver_లోపల_range(uint32_t driver_సంస్కరణ, uint32_t క్రింది_major, uint32_t క్రింది_minor, uint32_t పై_major, uint32_t పై_minor);

// shaders
శూన్య ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_fa_tuning_params get_fa_tuning_params(స్థిరరాశి vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_అడ్డువరుసలు, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc);
vk_fa_pipeline_state get_fa_pipeline_state(స్థిరరాశి vk_device& device, స్థిరరాశి vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc, bool use_ముసుగు, bool use_ముసుగు_opt, bool use_logit_softcap, bool use_sparse, ggml_type k_type, ggml_type v_type);
uint32_t get_subgroup_కొలత(స్థిరరాశి std::string &pipeline_name, స్థిరరాశి vk_device_architecture &arch);
శూన్య ggml_vk_ఎక్కించు_shaders(vk_device& device, vk_pipeline requested = nullptr);
bool ggml_vk_flash_attn_scalar_shmem_support(స్థిరరాశి vk_device& device, స్థిరరాశి vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type);
bool ggml_vk_flash_attn_coopmat_shmem_support(స్థిరరాశి vk_device& device, స్థిరరాశి vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type = GGML_TYPE_F16, ggml_type v_type = GGML_TYPE_F16);

// buffers
vk_బఫరు ggml_vk_create_బఫరు_check(vk_device& device, size_t కొలత, vk::జ్ఞాపకం_Property_Flags req_flags, vk::జ్ఞాపకం_Property_Flags fallback_flags = vk::జ్ఞాపకం_Property_Flags(0));
vk_బఫరు ggml_vk_create_బఫరు_device(vk_device& device, size_t కొలత);
శూన్య ggml_vk_destroy_బఫరు(vk_బఫరు& buf);
శూన్య * ggml_vk_host_malloc(size_t కొలత);
శూన్య ggml_vk_host_free(శూన్య * ptr);
శూన్య ggml_vk_host_get(స్థిరరాశి vk_device& device, స్థిరరాశి శూన్య * ptr, vk_బఫరు& buf, size_t& buf_స్థానభ్రంశం);
శూన్య ggml_vk_ensure_sync_staging_బఫరు(vk_device& device, size_t కొలత);
శూన్య ggml_vk_ensure_sync_staging_బఫరు(ggml_backend_vk_context * ctx, size_t కొలత);
bool ggml_vk_బఫరు_write_2d_async(vk_context subctx, vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు, bool sync_staging = false);
bool ggml_vk_బఫరు_write_async(vk_context subctx, vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t కొలత, bool sync_staging = false);
శూన్య ggml_vk_బఫరు_write_2d(vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు);
శూన్య ggml_vk_బఫరు_write(vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t కొలత);
bool ggml_vk_బఫరు_read_2d_async(vk_context subctx, vk_బఫరు& src, size_t స్థానభ్రంశం, శూన్య * dst, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు, bool sync_staging = false);
శూన్య ggml_vk_బఫరు_read_2d(vk_బఫరు& src, size_t స్థానభ్రంశం, శూన్య * dst, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు);
శూన్య ggml_vk_బఫరు_read(vk_బఫరు& src, size_t స్థానభ్రంశం, శూన్య * dst, size_t కొలత);
శూన్య ggml_vk_బఫరు_copy_async(vk_context& ctx, vk_బఫరు& dst, size_t dst_స్థానభ్రంశం, vk_బఫరు& src, size_t src_స్థానభ్రంశం, size_t కొలత);
శూన్య ggml_vk_బఫరు_copy(vk_బఫరు& dst, size_t dst_స్థానభ్రంశం, vk_బఫరు& src, size_t src_స్థానభ్రంశం, size_t కొలత);
శూన్య ggml_vk_బఫరు_memset_async(vk_context& ctx, vk_బఫరు& dst, size_t స్థానభ్రంశం, uint32_t c, size_t కొలత);
శూన్య ggml_vk_బఫరు_memset(vk_బఫరు& dst, size_t స్థానభ్రంశం, uint32_t c, size_t కొలత);
vk_బఫరు ggml_vk_బఫరు_from_host_ptr(vk_device & device, శూన్య * ptr, size_t కొలత);

// pipelines
uint64_t vk_tensor_స్థానభ్రంశం(స్థిరరాశి ggml_tensor * tensor);
uint32_t get_misalign_బైట్లు(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * t);
శూన్య ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx);
శూన్య ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n);
శూన్య ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx);
శూన్య ggml_vk_submit(vk_context& ctx, vk::Fence fence);
uint32_t ggml_vk_కనుగొను_వరుస_family_సూచీ(std::సదిశ<vk::వరుస_Family_Properties>& వరుస_family_props, స్థిరరాశి vk::వరుస_Flags& required, స్థిరరాశి vk::వరుస_Flags& avoid, int32_t compute_సూచీ, uint32_t min_అంకె_queues);
std::unique_ptr<vk_వరుస> ggml_vk_create_వరుస(vk_device& device, uint32_t వరుస_family_సూచీ, uint32_t వరుస_సూచీ, vk::PipelineStageFlags&& stage_flags, bool transfer_only);
std::unique_ptr<vk_వరుస> ggml_vk_create_aliased_వరుస(vk_device& device, స్థిరరాశి std::unique_ptr<vk_వరుస>& ఆధారం);
vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p);
vk_context ggml_vk_create_temporary_context(vk_command_pool& p);
శూన్య ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
శూన్య ggml_vk_వరుస_command_pools_cleanup(vk_device& device);
vk_subbuffer ggml_vk_subbuffer(స్థిరరాశి ggml_backend_vk_context* ctx, స్థిరరాశి vk_బఫరు& buf, size_t స్థానభ్రంశం = 0);
శూన్య ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx);
శూన్య ggml_vk_సమితి_event(vk_context& ctx, vk::Event& event);
శూన్య ggml_vk_wait_events(vk_context& ctx, std::సదిశ<vk::Event>&& events);
vk_subbuffer ggml_vk_tensor_subbuffer(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * tensor, bool allow_misalign = false);
శూన్య ggml_vk_cmd_label_begin(vk::Command_బఫరు buf, స్థిరరాశి అక్షరం * name);
శూన్య ggml_vk_ctx_చివర(vk_context& ctx);
శూన్య ggml_vk_ctx_begin(vk_device& device, vk_context& subctx);
vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx);
vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx);
bool ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx);
size_t ggml_vk_align_కొలత(size_t వెడల్పు, size_t align);
శూన్య deferred_memcpy(శూన్య * dst, స్థిరరాశి శూన్య * src, size_t కొలత, std::సదిశ<vk_staging_memcpy>* memcpys = nullptr);
శూన్య deferred_memset(శూన్య * dst, uint32_t val, size_t కొలత, std::సదిశ<vk_staging_memset>* memsets = nullptr);

// matmul
vk_pipeline ggml_vk_get_గమ్యం_fp16(ggml_backend_vk_context * ctx, ggml_type type);
శూన్య ggml_vk_matmul(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& విభజించు_k_బఫరు, uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d, uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d, uint32_t విభజించు_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3, uint32_t padded_n, uint64_t b_నిలువు_బైట్లు, uint64_t d_నిలువు_బైట్లు);
bool ggml_vk_dim01_contiguous(స్థిరరాశి ggml_tensor * tensor);
vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * src, స్థిరరాశి ggml_tensor * dst, ggml_type గమ్యం);
vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type);
శూన్య ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి vk_subbuffer & లోపల, స్థిరరాశి vk_subbuffer & బహిర్గతం, uint32_t ne);
శూన్య ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * mixes, స్థిరరాశి ggml_tensor * scale, స్థిరరాశి ggml_tensor * పునాది, ggml_tensor * dst);
శూన్య ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * x, స్థిరరాశి ggml_tensor * weights, ggml_tensor * dst);
శూన్య ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * x, స్థిరరాశి ggml_tensor * residual, స్థిరరాశి ggml_tensor * post, స్థిరరాశి ggml_tensor * comb, ggml_tensor * dst, స్థిరరాశి ggml_tensor * gate_scale_లోపల = nullptr);
శూన్య ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
bool ggml_vk_use_mul_mat_vec_id(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);

// flash-attn
శూన్య ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * q, స్థిరరాశి ggml_tensor * k, స్థిరరాశి ggml_tensor * v, స్థిరరాశి ggml_tensor * ముసుగు, స్థిరరాశి ggml_tensor * sinks, ggml_tensor * dst);

// operators
శూన్య ggml_vk_cpy_గమ్యం_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి vk_subbuffer & లోపల, స్థిరరాశి vk_subbuffer & బహిర్గతం);
bool ggml_vk_can_use_fwht(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * dst);
శూన్య ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst);
శూన్య ggml_vk_get_అడ్డువరుసలు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_get_అడ్డువరుసలు_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_multi_కూడు(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_కూడు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_బహిర్గతం_prod(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
పూర్ణాంకం ggml_vk_unary_mul_op_సూచీ(ggml_unary_op op);
శూన్య ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_కూడు_id(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, ggml_tensor * dst);
శూన్య ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_gated_రేఖీయ_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_ssm_పరిశీలన(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_opt_అడుగు_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_opt_అడుగు_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, ggml_tensor * dst);
శూన్య ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_repeat_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_సమితి_అడ్డువరుసలు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_silu_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_సమూహం_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
uint32_t ggml_vk_rms_partials_కొలత(ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor *కణుపు);
శూన్య ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, దశాంశం * op_params);
శూన్య ggml_vk_rms_norm_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_diag_ముసుగు_inf(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * src2, ggml_tensor * dst);
శూన్య ggml_vk_soft_max_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, bool backprop);
శూన్య ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_కూడిక(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_కూడిక_అడ్డువరుసలు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_సగటు(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_cross_entropy_loss_వెనుక(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst);
శూన్య ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_లెక్క_equal(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);
శూన్య ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst);
శూన్య ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, స్థిరరాశి ggml_tensor * src0, ggml_tensor * dst);

//లేఖాచిత్రం
శూన్య ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx);
bool ggml_vk_build_లేఖాచిత్రం(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, ggml_tensor *కణుపు_begin, పూర్ణాంకం కణుపు_idx_begin, bool చివరి_కణుపు, bool almost_ready, bool submit);
శూన్య ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, పూర్ణాంకం tensor_idx, bool almost_ready);
శూన్య ggml_vk_లేఖాచిత్రం_cleanup(ggml_backend_vk_context * ctx);
శూన్య ggml_vk_cleanup(ggml_backend_vk_context * ctx);
శూన్య ggml_vk_synchronize(ggml_backend_vk_context * ctx);
bool ggml_vk_is_ఖాళీలు(ggml_tensor * కణుపు);
bool ggml_vk_can_fuse(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, std::initializer_list<గణన ggml_op> ops);
bool ggml_vk_can_fuse_ssm_conv(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, పూర్ణాంకం అంకె_extra);
bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, topk_moe_బహుళకం బహుళకం);
bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
bool ggml_vk_can_fuse_rope_సమితి_అడ్డువరుసలు(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
bool ggml_vk_can_fuse_rms_norm_సమితి_అడ్డువరుసలు(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
bool ggml_vk_tensors_overlap(స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b, bool elementwise);
bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
uint32_t ggml_vk_fuse_multi_కూడు(ggml_backend_vk_context * ctx, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx);
శూన్య ggml_vk_లేఖాచిత్రం_optimize(ggml_backend_t backend, నిర్మాణం ggml_cgraph * లేఖాచిత్రం, నిర్మాణం ggml_backend_లేఖాచిత్రం_optimize_params * params);

// backend
bool ggml_backend_బఫరు_is_vk(ggml_backend_బఫరు_t బఫరు);
శూన్య ggml_backend_vk_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు);
శూన్య * ggml_backend_vk_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు);
గణన ggml_స్థితి ggml_backend_vk_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor);
శూన్య ggml_backend_vk_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t విలువ, size_t స్థానభ్రంశం, size_t కొలత);
శూన్య ggml_backend_vk_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
శూన్య ggml_backend_vk_బఫరు_సమితి_tensor_2d(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం);
శూన్య ggml_backend_vk_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
శూన్య ggml_backend_vk_బఫరు_get_tensor_2d(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం);
bool ggml_backend_vk_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst);
శూన్య ggml_backend_vk_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t విలువ);
స్థిరరాశి అక్షరం * ggml_backend_vk_బఫరు_type_name(ggml_backend_బఫరు_type_t buft);
ggml_backend_బఫరు_t ggml_backend_vk_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t కొలత);
size_t ggml_backend_vk_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft);
size_t ggml_backend_vk_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft);
size_t ggml_backend_vk_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor);
శూన్య ggml_backend_vk_free(ggml_backend_t backend);
ggml_backend_reg_t ggml_backend_vk_reg();

// debug
int64_t ggml_vk_get_op_batch_కొలత(స్థిరరాశి ggml_tensor * op);
#నిర్వచితమైతే GGML_VULKAN_నిరంతరం_TESTS
మూస <typename X_TYPE, typename Y_TYPE>
శూన్య ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t అంకె_it, పూర్ణాంకం విభజించు_k, పూర్ణాంకం shader_కొలత);
శూన్య ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant);
శూన్య ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t అంకె_it, size_t విభజించు_k, size_t shader_కొలత, ggml_type quant, bool mmq = false);
#అయితే_ముగింపు
#నిర్వచితమైతే GGML_VULKAN_CHECK_RESULTS
బాహ్య size_t vk_skip_checks;
బాహ్య size_t vk_output_tensor;
శూన్య ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, పూర్ణాంకం tensor_idx);
శూన్య ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, పూర్ణాంకం tensor_idx);
#అయితే_ముగింపు

// ggml-vulkan.cpp (residual)
bool ggml_vk_lightning_indexer_k_type_supported(ggml_type type);
శూన్య ggml_vk_print_device_fault_info(స్థిరరాశి vk_device& device);
uint64_t ggml_vk_get_కణుపు_flops(స్థిరరాశి ggml_tensor * కణుపు);
శూన్య ggml_vk_print_కణుపు_list(స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం ప్రారంభం, పూర్ణాంకం చివర);
శూన్య ggml_vk_print_device_కోల్పోయిన_info(స్థిరరాశి vk_device& device);
size_t ggml_vk_tensor_బఫరు_స్థానభ్రంశం(స్థిరరాశి ggml_backend_vk_context * ctx, స్థిరరాశి ggml_tensor * t);
size_t ggml_vk_descriptor_స్థానభ్రంశం(size_t tensor_స్థానభ్రంశం, size_t alignment, size_t type_కొలత);
uint32_t ggml_vk_concat_unit_కొలత(ggml_type type);
bool ggml_vk_concat_supported(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * dst);

మూస <typename T>
అంతఃవరుస శూన్య ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::Descriptor_బఫరు_Info> స్థిరరాశి& descriptor_బఫరు_infos, స్థిరరాశి T &నెట్టు_constants, std::array<uint32_t, 3> elements) {
    స్థిరరాశి uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    స్థిరరాశి uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    స్థిరరాశి uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    కొరకు (స్వయంచాలక& బఫరు : descriptor_బఫరు_infos) {
        std::cerr << "(" << బఫరు.బఫరు << ", " << బఫరు.స్థానభ్రంశం << ", " << బఫరు.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[0] &&
                wg1 <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[1] &&
                wg2 <= ctx->device->properties.limits.max_Compute_Work_సమూహం_లెక్క[2]);
    GGML_ASSERT(ctx->descriptor_సమితి_idx < ctx->descriptor_sets.కొలత());
    GGML_ASSERT(descriptor_బఫరు_infos.కొలత() <= MAX_PARAMETER_లెక్క);
    GGML_ASSERT(pipeline->parameter_లెక్క == descriptor_బఫరు_infos.కొలత());
    GGML_ASSERT(pipeline->నెట్టు_constant_కొలత == నెట్టు_constant_కొలత(నెట్టు_constants));

    స్థిరరాశి uint32_t descriptor_సమితి_idx = ctx->descriptor_సమితి_idx++;
    vk::Descriptor_సమితి& descriptor_సమితి = ctx->descriptor_sets[descriptor_సమితి_idx];

    // a new buffer can get the handle of a destroyed one, so drop all cached bindings after any destroy
    స్థిరరాశి uint64_t destroy_లెక్క = ctx->device->బఫరు_destroy_లెక్క.ఎక్కించు(std::memory_order_acquire);
    ఒకవేళ (ctx->descriptor_సమితి_bindings_destroy_లెక్క != destroy_లెక్క) {
        కొరకు (స్వయంచాలక & b : ctx->descriptor_సమితి_bindings) {
            b.clear();
        }
        ctx->descriptor_సమితి_bindings_destroy_లెక్క = destroy_లెక్క;
    }

    // skip the write if this set already holds these bindings from the last graph
    std::సదిశ<vk::Descriptor_బఫరు_Info> & bindings = ctx->descriptor_సమితి_bindings[descriptor_సమితి_idx];
    bool same = !ctx->device->disable_descriptor_reuse && bindings.కొలత() == descriptor_బఫరు_infos.కొలత();
    ఒకవేళ (same) {
        size_t i = 0;
        కొరకు (స్థిరరాశి vk::Descriptor_బఫరు_Info & info : descriptor_బఫరు_infos) {
            స్థిరరాశి vk::Descriptor_బఫరు_Info & prev = bindings[i++];
            ఒకవేళ (prev.బఫరు != info.బఫరు || prev.స్థానభ్రంశం != info.స్థానభ్రంశం || prev.range != info.range) {
                same = false;
                ఆపు;
            }
        }
    }
    ఒకవేళ (!same) {
        vk::Write_Descriptor_సమితి write_descriptor_సమితి{ descriptor_సమితి, 0, 0, pipeline->parameter_లెక్క, vk::DescriptorType::e_Storage_బఫరు, nullptr, descriptor_బఫరు_infos.begin() };
        ctx->device->device.updateDescriptorSets({ write_descriptor_సమితి }, {});
        bindings.కేటాయించు(descriptor_బఫరు_infos.begin(), descriptor_బఫరు_infos.చివర());
    }

    subctx->s->బఫరు->buf.నెట్టు_Constants(pipeline->layout, vk::Shader_Stage_Flag_బిట్లు::eCompute, 0, నెట్టు_constant_కొలత(నెట్టు_constants), నెట్టు_constant_దత్తాంశం(నెట్టు_constants));
    subctx->s->బఫరు->buf.bindPipeline(vk::Pipeline_Bind_బిందువు::eCompute, pipeline->pipeline);
    subctx->s->బఫరు->buf.bindDescriptorSets(vk::Pipeline_Bind_బిందువు::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_సమితి },
                                {});
    {
        ggml_vk_debug_label dbg(subctx, pipeline->name, wg0, wg1, wg2);
        subctx->s->బఫరు->buf.dispatch(wg0, wg1, wg2);
    }
}

