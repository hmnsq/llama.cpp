#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan-common.h"

ਤਾਰਕਿਕ vk_ਸਮ੍ਰਿਤੀ_logger_enabled = ਝੂਠ;

ਤਾਰਕਿਕ vk_perf_logger_enabled = ਝੂਠ;

ਤਾਰਕਿਕ vk_perf_logger_concurrent = ਝੂਠ;

ਤਾਰਕਿਕ vk_enable_sync_logger = ਝੂਠ;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

ਖਾਲੀ vk_ਸਮ੍ਰਿਤੀ_logger::log_ਵੰਡ(vk_ਬਫਰ_ref buf_ref, size_t size) {
    ਜੇ (!vk_ਸਮ੍ਰਿਤੀ_logger_enabled) {
        ਵਾਪਸ;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_ਬਫਰ buf = buf_ref.ਤਾਲਾ();
    ਅਚਲ ਤਾਰਕਿਕ device = ਤਾਰਕਿਕ(buf->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal);
    ਅਚਲ std::string type = device ? "device" : "host";
    allocations[buf->ਬਫਰ] = size;
    ਕੁੱਲ_device += device ? size : 0;
    ਕੁੱਲ_host += device ? 0 : size;
    VK_LOG_ਸਮ੍ਰਿਤੀ(buf->device->name << ": +" << format_ਮਾਪ(size) << " " << type << " at " << buf->ਬਫਰ << ". Total device: " << format_ਮਾਪ(ਕੁੱਲ_device) << ", total host: " << format_ਮਾਪ(ਕੁੱਲ_host));
}

ਖਾਲੀ vk_ਸਮ੍ਰਿਤੀ_logger::log_deallocation(vk_ਬਫਰ_ref buf_ref) {
    ਜੇ (buf_ref.expired() || buf_ref.ਤਾਲਾ()->size == 0 || !vk_ਸਮ੍ਰਿਤੀ_logger_enabled) {
        ਵਾਪਸ;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_ਬਫਰ buf = buf_ref.ਤਾਲਾ();
    ਅਚਲ ਤਾਰਕਿਕ device = ਤਾਰਕਿਕ(buf->ਸਮ੍ਰਿਤੀ_property_flags & vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal);
    std::string type = device ? "device" : "host";
    ਸਵੈਚਾਲਿਤ it = allocations.find(buf->ਬਫਰ);
    ਜੇ (it != allocations.end()) {
        ਕੁੱਲ_device -= device ? it->second : 0;
        ਕੁੱਲ_host -= device ? 0 : it->second;
        VK_LOG_ਸਮ੍ਰਿਤੀ(buf->device->name << ": -" << format_ਮਾਪ(it->second) << " " << type << " at " << buf->ਬਫਰ << ". Total device: " << format_ਮਾਪ(ਕੁੱਲ_device) << ", total host: " << format_ਮਾਪ(ਕੁੱਲ_host));
        allocations.erase(it);
    } ਨਹੀਂਤਾਂ {
        VK_LOG_ਸਮ੍ਰਿਤੀ("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->ਬਫਰ);
    }
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

ਸਥਿਰ ਖਾਲੀ ggml_vk_print_tensor(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ਅੱਖਰ * name);
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_ਲੜੀਵਾਰ_TESTS
ਸਥਿਰ ਖਾਲੀ ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, ggml_type type, ਪੂਰਨ ne0, ਪੂਰਨ ne1, ਪੂਰਨ i0, ਪੂਰਨ i1, ਪੂਰਨ i2) {
    ਜੇ (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        ਵਾਪਸ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    ਲਈ (ਪੂਰਨ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ਲਈ (ਪੂਰਨ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ਲਈ (ਪੂਰਨ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ਜੇ (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                ਦਸ਼ਮਲਵ val;
                ਜੇ (type == GGML_TYPE_F32) {
                    val = *((ਅਚਲ ਦਸ਼ਮਲਵ *) ਅੰਕੜੇ + i2*ne1*ne0 + idx1*ne0 + idx0);
                } ਨਹੀਂਤਾਂ ਜੇ (type == GGML_TYPE_F16) {
                    val = ggml_fp16_ਮੰਜ਼ਿਲ_fp32(*((ਅਚਲ ggml_fp16_t *) ਅੰਕੜੇ + i2*ne1*ne0 + idx1*ne0 + idx0));
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ਨਹੀਂਤਾਂ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ X_TYPE, ਕਿਸਮ_ਨਾਂ Y_TYPE>
ਖਾਲੀ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ਅੰਕ_it, ਪੂਰਨ ਵੰਡੋ_k, ਪੂਰਨ shader_ਮਾਪ) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << ਅੰਕ_it << ", " << ਵੰਡੋ_k << ", " << shader_ਮਾਪ << ")");
    ਅਚਲ size_t x_ne = m * k * batch;
    ਅਚਲ size_t y_ne = k * n * batch;
    ਅਚਲ size_t d_ne = m * n * batch;

    ggml_type x_type = std::is_same<ਦਸ਼ਮਲਵ, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<ਦਸ਼ਮਲਵ, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_ਕੁੰਜੀ mm_test_ਕੁੰਜੀ{x_type, y_type, ਝੂਠ, ਝੂਠ};
    ਸਵੈਚਾਲਿਤ mm_test_it = ctx->device->pipeline_matmul.find(mm_test_ਕੁੰਜੀ);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.ਖਾਲੀ_ਥਾਵਾਂ());
    ਸਵੈਚਾਲਿਤ& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_ਮਾਪ >= 0 && shader_ਮਾਪ < (ਪੂਰਨ)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_ਮਾਪ);
    vk_pipeline p = mm_test_configs[shader_ਮਾਪ].aligned ? mm_test_configs[shader_ਮਾਪ].aligned : mm_test_configs[shader_ਮਾਪ].unaligned;

    ਅਚਲ size_t kpad = ggml_vk_align_ਮਾਪ(k, mm_test_configs[shader_ਮਾਪ].align);

    ਜੇ (k != kpad) {
        p = mm_test_configs[shader_ਮਾਪ].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_ਮਾਪ);
    }

    ਜੇ (ਵੰਡੋ_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ਵੰਡੋ_k_reduce, ਅੰਕ_it);

        ਜੇ (ctx->prealloc_ਵੰਡੋ_k == ਖਾਲੀ_ਸੂਚਕ || ctx->prealloc_ਵੰਡੋ_k->size < ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k) {
            // Resize buffer
            ਜੇ (ctx->prealloc_ਵੰਡੋ_k != ਖਾਲੀ_ਸੂਚਕ) {
                ggml_vk_destroy_ਬਫਰ(ctx->prealloc_ਵੰਡੋ_k);
            }
            ctx->prealloc_ਵੰਡੋ_k = ggml_vk_create_ਬਫਰ_check(ctx->device, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_ਬਫਰ d_X = ggml_vk_create_ਬਫਰ_check(ctx->device, ਆਕਾਰ(X_TYPE) * x_ne, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    vk_ਬਫਰ d_Y = ggml_vk_create_ਬਫਰ_check(ctx->device, ਆਕਾਰ(Y_TYPE) * y_ne, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    vk_ਬਫਰ d_D = ggml_vk_create_ਬਫਰ_check(ctx->device, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(Y_TYPE) * y_ne);
    ਦਸ਼ਮਲਵ* d = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne);

    ਲਈ (size_t i = 0; i < x_ne; i++) {
        ਜੇ (std::is_same<ਦਸ਼ਮਲਵ, X_TYPE>()) {
            x[i] = (rand() / (ਦਸ਼ਮਲਵ)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } ਨਹੀਂਤਾਂ ਜੇ (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_ਮੰਜ਼ਿਲ_fp16((rand() / (ਦਸ਼ਮਲਵ)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("fatal error");
        }
    }
    ਲਈ (size_t i = 0; i < y_ne; i++) {
        ਜੇ (std::is_same<ਦਸ਼ਮਲਵ, Y_TYPE>()) {
            y[i] = (rand() / (ਦਸ਼ਮਲਵ)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } ਨਹੀਂਤਾਂ ਜੇ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_ਮੰਜ਼ਿਲ_fp16((rand() / (ਦਸ਼ਮਲਵ)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_ਬਫਰ_write(d_X, 0, x, ਆਕਾਰ(X_TYPE) * k * m * batch);
    ggml_vk_ਬਫਰ_write(d_Y, 0, y, ਆਕਾਰ(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ਲਈ (size_t i = 0; i < ਅੰਕ_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_ਵੰਡੋ_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            ਵੰਡੋ_k, batch, batch, batch, 1, 1, n,
            k * ਆਕਾਰ(Y_TYPE), m * ਆਕਾਰ(ਦਸ਼ਮਲਵ)
        );
    }
    ggml_vk_ctx_ਸਿਰਾ(subctx);

    ਸਵੈਚਾਲਿਤ begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ਸੱਚ, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_ਕਤਾਰ_command_pools_cleanup(ctx->device);

    ਸਵੈਚਾਲਿਤ end = std::chrono::high_resolution_clock::now();
    ਦੁੱਗਣਾ ਸਮਾਂ = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_ਬਫਰ_read(d_D, 0, d, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne);

    ਦਸ਼ਮਲਵ * d_chk = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    ਜੇ (std::is_same<ਦਸ਼ਮਲਵ, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } ਨਹੀਂਤਾਂ ਜੇ (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }
    ਜੇ (std::is_same<ਦਸ਼ਮਲਵ, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } ਨਹੀਂਤਾਂ ਜੇ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->ਅੰਕੜੇ = x;
    src1_ggml->ਅੰਕੜੇ = y;
    tensor_ggml->ਅੰਕੜੇ = d_chk;

    ggml_cgraph * cgraph = ggml_new_ਗ੍ਰਾਫ(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_ਗ੍ਰਾਫ_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ਦੁੱਗਣਾ avg_err = 0.0;
    ਪੂਰਨ ਪਹਿਲਾ_err_n = -1;
    ਪੂਰਨ ਪਹਿਲਾ_err_m = -1;
    ਪੂਰਨ ਪਹਿਲਾ_err_b = -1;

    ਲਈ (size_t i = 0; i < m*n*batch; i++) {
        ਦੁੱਗਣਾ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ਜੇ ((err > 0.05f || std::isnan(err)) && ਪਹਿਲਾ_err_n == -1) {
            ਪਹਿਲਾ_err_b = i / (m * n);
            ਪਹਿਲਾ_err_n = (i % (m * n)) / m;
            ਪਹਿਲਾ_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ਦੁੱਗਣਾ tflops = 2.0*m*n*k*batch*ਅੰਕ_it / (ਸਮਾਂ / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << ਵੰਡੋ_k << " matmul " << ਸਮਾਂ / ਅੰਕ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::ਸਤਰ_ਅੰਤ;

    ਜੇ (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << ਪਹਿਲਾ_err_m << " n = " << ਪਹਿਲਾ_err_n << " b = " << ਪਹਿਲਾ_err_b << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "Actual result: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(d, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);
        std::cerr << "Expected result: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(d_chk, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

        ਜੇ (ਵੰਡੋ_k > 1) {
            ਦਸ਼ਮਲਵ * ਵੰਡੋ_k_buf = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k);
            ggml_vk_ਬਫਰ_read(ctx->prealloc_ਵੰਡੋ_k, 0, ਵੰਡੋ_k_buf, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k);

            std::cerr << "d_buf0: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            std::cerr << "d_buf1: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf + d_ne, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            std::cerr << "d_buf2: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            std::cerr << "d_buf3: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            ਮੁਕਤ(ਵੰਡੋ_k_buf);
        }
    }

    ਮੁਕਤ(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_ਬਫਰ(d_X);
    ggml_vk_destroy_ਬਫਰ(d_Y);
    ggml_vk_destroy_ਬਫਰ(d_D);

    ਮੁਕਤ(x);
    ਮੁਕਤ(y);
    ਮੁਕਤ(d);
}

ਸਾਂਚਾ ਖਾਲੀ ggml_vk_test_matmul<ggml_fp16_t, ਦਸ਼ਮਲਵ>(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ਅੰਕ_it, ਪੂਰਨ ਵੰਡੋ_k, ਪੂਰਨ shader_ਮਾਪ);

ਸਥਿਰ ਖਾਲੀ ggml_vk_print_tensor_area(ਅਚਲ ggml_tensor * tensor, ਪੂਰਨ i0, ਪੂਰਨ i1, ਪੂਰਨ i2, ਪੂਰਨ i3) {
    ਜੇ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        ਵਾਪਸ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    ਲਈ (ਪੂਰਨ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ਲਈ (ਪੂਰਨ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ਲਈ (ਪੂਰਨ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ਜੇ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                ਦਸ਼ਮਲਵ val;
                ਜੇ (tensor->type == GGML_TYPE_F32) {
                    val = *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ਨਹੀਂਤਾਂ ਜੇ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_ਮੰਜ਼ਿਲ_fp32(*(ggml_fp16_t *) ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ਨਹੀਂਤਾਂ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_quantize_ਅੰਕੜੇ(ਅਚਲ ਦਸ਼ਮਲਵ * from, ਖਾਲੀ * ਮੰਜ਼ਿਲ, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, ਮੰਜ਼ਿਲ, 0, 1, ne, ਖਾਲੀ_ਸੂਚਕ);
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_dequantize_ਅੰਕੜੇ(ਅਚਲ ਖਾਲੀ * from, ਦਸ਼ਮਲਵ * ਮੰਜ਼ਿਲ, size_t ne, ggml_type quant) {
    ਜੇ (quant == GGML_TYPE_F32) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਮੰਜ਼ਿਲ, from, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * ne);
        ਵਾਪਸ;
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ * tt = ggml_get_type_traits(quant);

    ggml_ਮੰਜ਼ਿਲ_float_t dequant_fn = tt->ਮੰਜ਼ਿਲ_float;

    dequant_fn(from, ਮੰਜ਼ਿਲ, ne);
}

ਖਾਲੀ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    ਅਚਲ size_t x_sz = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * ne;
    ਅਚਲ size_t x_sz_f16 = ਆਕਾਰ(ggml_fp16_t) * ne;
    ਅਚਲ size_t qx_sz = ne * ggml_type_ਮਾਪ(quant)/ggml_blck_ਮਾਪ(quant);
    ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(x_sz);
    ਖਾਲੀ * qx = ਸਮ੍ਰਿਤੀ_ਦਿਓ(qx_sz);
    vk_ਬਫਰ qx_buf = ggml_vk_create_ਬਫਰ_check(ctx->device, qx_sz, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    vk_ਬਫਰ x_buf = ggml_vk_create_ਬਫਰ_check(ctx->device, x_sz_f16, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    ਦਸ਼ਮਲਵ * x_ref = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(x_sz_f16);

    ਲਈ (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (ਦਸ਼ਮਲਵ)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_ਮੰਜ਼ਿਲ_fp16(ctx, quant);

    ggml_vk_quantize_ਅੰਕੜੇ(x, qx, ne, quant);
    ggml_vk_dequantize_ਅੰਕੜੇ(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_ਬਫਰ_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ਅਚਲ std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_ਸਿਰਾ(subctx);

    ਸਵੈਚਾਲਿਤ begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ਸੱਚ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_ਕਤਾਰ_command_pools_cleanup(ctx->device);

    ਸਵੈਚਾਲਿਤ end = std::chrono::high_resolution_clock::now();

    ਦੁੱਗਣਾ ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_ਬਫਰ_read(x_buf, 0, x_chk, x_sz_f16);

    ਪੂਰਨ ਪਹਿਲਾ_err = -1;

    ਦੁੱਗਣਾ avg_err = 0.0;
    ਲਈ (size_t i = 0; i < ne; i++) {
        ਦੁੱਗਣਾ error = std::fabs(x_ref[i] - ggml_fp16_ਮੰਜ਼ਿਲ_fp32(x_chk[i]));
        avg_err += error;

        ਜੇ (ਪਹਿਲਾ_err < 0 && error > 0.05) {
            ਪਹਿਲਾ_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::ਸਤਰ_ਅੰਤ;

    ਜੇ (avg_err > 0.1) {
        std::cerr << "first_error = " << ਪਹਿਲਾ_err << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "Actual result: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ਲਈ (ਪੂਰਨ i = std::max(0, ਪਹਿਲਾ_err - 5); i < std::min((ਪੂਰਨ)ne, ਪਹਿਲਾ_err + 5); i++) {
            std::cerr << ggml_fp16_ਮੰਜ਼ਿਲ_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::ਸਤਰ_ਅੰਤ << "Expected result: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ਲਈ (ਪੂਰਨ i = std::max(0, ਪਹਿਲਾ_err - 5); i < std::min((ਪੂਰਨ)ne, ਪਹਿਲਾ_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::ਸਤਰ_ਅੰਤ;
    }

    ggml_vk_destroy_ਬਫਰ(x_buf);
    ggml_vk_destroy_ਬਫਰ(qx_buf);

    ਮੁਕਤ(x);
    ਮੁਕਤ(qx);
    ਮੁਕਤ(x_ref);
    ਮੁਕਤ(x_chk);
}

// This does not work without ggml q8_1 quantization support
//
// typedef uint16_t ggml_half;
// typedef uint32_t ggml_half2;
//
// #define QK8_1 32
// typedef struct {
//     union {
//         struct {
//             ggml_half d; // delta
//             ggml_half s; // d * sum(qs[i])
//         } GGML_COMMON_AGGR_S;
//         ggml_half2 ds;
//     } GGML_COMMON_AGGR_U;
//     int8_t qs[QK8_1]; // quants
// } block_q8_1;
//
// static void ggml_vk_test_quantize(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
//     VK_LOG_DEBUG("ggml_vk_test_quantize(" << ne << ")");
//     GGML_ASSERT(quant == GGML_TYPE_Q8_1);
//
//     const size_t x_sz = sizeof(float) * ne;
//     const size_t qx_sz = ne * ggml_type_size(quant)/ggml_blck_size(quant);
//     float * x = (float *) malloc(x_sz);
//     block_q8_1 * qx     = (block_q8_1 *)malloc(qx_sz);
//     block_q8_1 * qx_res = (block_q8_1 *)malloc(qx_sz);
//     vk_buffer x_buf = ggml_vk_create_buffer_check(ctx->device, x_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//     vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//
//     for (size_t i = 0; i < ne; i++) {
//         x[i] = rand() / (float)RAND_MAX;
//     }
//
//     vk_pipeline p = ggml_vk_get_quantize_pipeline(ctx, quant);
//
//     ggml_pipeline_request_descriptor_sets(ctx, p, 1);
//
//     ggml_pipeline_allocate_descriptor_sets(ctx);
//
//     ggml_vk_buffer_write(x_buf, 0, x, x_sz);
//
//     vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
//     ggml_vk_ctx_begin(ctx->device, subctx);
//     ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, x_buf), ggml_vk_subbuffer(ctx, qx_buf), ne);
//     ggml_vk_ctx_end(subctx);
//
//     auto begin = std::chrono::high_resolution_clock::now();
//
//     ggml_vk_submit(subctx, ctx->fence);
//     VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_quantize waitForFences");
//     ctx->device->device.resetFences({ ctx->fence });
//     ggml_vk_queue_command_pools_cleanup(ctx->device);
//
//     auto end = std::chrono::high_resolution_clock::now();
//
//     double ms_quant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
//     ggml_vk_buffer_read(qx_buf, 0, qx, qx_sz);
//
//     ggml_vk_quantize_data(x, qx_res, ne, quant);
//
//     int first_err = -1;
//
//     for (size_t i = 0; i < ne / 32; i++) {
//         double error = std::fabs(ggml_fp16_to_fp32(qx_res[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) - ggml_fp16_to_fp32(qx[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         error = std::fabs(ggml_fp16_to_fp32(qx_res[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s) - ggml_fp16_to_fp32(qx[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         for (size_t j = 0; j < 32; j++) {
//             uint64_t error = std::abs(qx_res[i].qs[j] - qx[i].qs[j]);
//
//             if (first_err < 0 && error > 1) {
//                 first_err = i;
//             }
//         }
//     }
//
//     std::cerr << "TEST QUANTIZE " << ggml_type_name(quant) << " time=" << ms_quant << "ms " << (first_err == -1 ? "CORRECT" : "INCORRECT") << std::endl;
//
//     if (first_err != -1) {
//         std::cerr << "first_error = " << first_err << std::endl;
//         std::cerr << "Actual result: " << std::endl << std::endl;
//         std::cout << "d=" << ggml_fp16_to_fp32(qx[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) << " s=" << ggml_fp16_to_fp32(qx[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl << std::endl << "Expected result: " << std::endl << std::endl;
//         std::cout << "d=" << ggml_fp16_to_fp32(qx_res[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) << " s=" << ggml_fp16_to_fp32(qx_res[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx_res[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl;
//     }
//
//     ggml_vk_destroy_buffer(x_buf);
//     ggml_vk_destroy_buffer(qx_buf);
//
//     free(x);
//     free(qx);
//     free(qx_res);
// }

ਖਾਲੀ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ਅੰਕ_it, size_t ਵੰਡੋ_k, size_t shader_ਮਾਪ, ggml_type quant, ਤਾਰਕਿਕ mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << ਅੰਕ_it << ", " << ਵੰਡੋ_k << ", " << ggml_type_name(quant) << ")");
    ਅਚਲ size_t x_ne = m * k * batch;
    ਅਚਲ size_t y_ne = k * n * batch;
    ਅਚਲ size_t d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    ਤਾਰਕਿਕ f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_ਕੁੰਜੀ dq_ਕੁੰਜੀ{quant, b_type, ਝੂਠ, f16acc};
    ਸਵੈਚਾਲਿਤ dq_it = ctx->device->pipeline_matmul.find(dq_ਕੁੰਜੀ);
    ਜੇ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਜੇ (f16acc) {
            dq_ਕੁੰਜੀ.f16acc = ਝੂਠ;
            dq_it = ctx->device->pipeline_matmul.find(dq_ਕੁੰਜੀ);
        }
    }
    ਜੇ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.ਖਾਲੀ_ਥਾਵਾਂ()) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::ਸਤਰ_ਅੰਤ;
        ਵਾਪਸ;
    }
    ਸਵੈਚਾਲਿਤ& dq_configs = dq_it->second;
    ਜੇ (shader_ਮਾਪ >= (ਪੂਰਨ)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_ਮਾਪ << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::ਸਤਰ_ਅੰਤ;
        ਵਾਪਸ;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_ਮਾਪ);
    vk_pipeline p = dq_configs[shader_ਮਾਪ].aligned ? dq_configs[shader_ਮਾਪ].aligned : dq_configs[shader_ਮਾਪ].unaligned;

    ਅਚਲ size_t kpad = mmq ? 0 : ggml_vk_align_ਮਾਪ(k, dq_configs[shader_ਮਾਪ].align);

    ਜੇ (mmq || k != kpad) {
        p = dq_configs[shader_ਮਾਪ].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_ਮਾਪ);
    }

    ਜੇ (p == ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::ਸਤਰ_ਅੰਤ;
        ਵਾਪਸ;
    }

    ਅਚਲ size_t x_sz = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * x_ne;
    ਅਚਲ size_t y_sz = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * y_ne;
    ਅਚਲ size_t qx_sz = x_ne * ggml_type_ਮਾਪ(quant)/ggml_blck_ਮਾਪ(quant);
    ਅਚਲ size_t qy_sz = mmq ? y_ne * ggml_type_ਮਾਪ(GGML_TYPE_Q8_1)/ggml_blck_ਮਾਪ(GGML_TYPE_Q8_1) : y_sz;
    ਅਚਲ size_t d_sz = ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne;
    ਦਸ਼ਮਲਵ * x = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(x_sz);
    ਦਸ਼ਮਲਵ * y = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(y_sz);
    ਖਾਲੀ * qx = ਸਮ੍ਰਿਤੀ_ਦਿਓ(qx_sz);
    vk_ਬਫਰ qx_buf = ggml_vk_create_ਬਫਰ_check(ctx->device, qx_sz, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    vk_ਬਫਰ y_buf = ggml_vk_create_ਬਫਰ_check(ctx->device, y_sz, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    vk_ਬਫਰ qy_buf = ggml_vk_create_ਬਫਰ_check(ctx->device, qy_sz, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    vk_ਬਫਰ d_buf = ggml_vk_create_ਬਫਰ_check(ctx->device, d_sz, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
    ਦਸ਼ਮਲਵ * d = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(d_sz);
    ਦਸ਼ਮਲਵ * d_chk = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(d_sz);

    ਲਈ (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (ਦਸ਼ਮਲਵ)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_ਅੰਕੜੇ(x, qx, x_ne, quant);

    ਲਈ (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (ਦਸ਼ਮਲਵ)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    ਜੇ (ਵੰਡੋ_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ਵੰਡੋ_k_reduce, ਅੰਕ_it);

        ਜੇ (ctx->prealloc_ਵੰਡੋ_k == ਖਾਲੀ_ਸੂਚਕ || ctx->prealloc_ਵੰਡੋ_k->size < ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k) {
            // Resize buffer
            ਜੇ (ctx->prealloc_ਵੰਡੋ_k != ਖਾਲੀ_ਸੂਚਕ) {
                ggml_vk_destroy_ਬਫਰ(ctx->prealloc_ਵੰਡੋ_k);
            }
            ctx->prealloc_ਵੰਡੋ_k = ggml_vk_create_ਬਫਰ_check(ctx->device, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k, {vk::ਸਮ੍ਰਿਤੀ_Property_Flag_ਬਿੱਟ::eDeviceLocal});
        }
    }
    ਜੇ (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, ਅੰਕ_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_ਬਫਰ_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_ਬਫਰ_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ਜੇ (mmq) {
        ਲਈ (size_t i = 0; i < ਅੰਕ_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_ਵੰਡੋ_k, 0, ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                ਵੰਡੋ_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_ਮਾਪ(GGML_TYPE_Q8_1)) * ggml_type_ਮਾਪ(GGML_TYPE_Q8_1), m * ਆਕਾਰ(ਦਸ਼ਮਲਵ)
            );
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (size_t i = 0; i < ਅੰਕ_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_ਵੰਡੋ_k, 0, ctx->prealloc_ਮਾਪ_ਵੰਡੋ_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                ਵੰਡੋ_k, batch, batch, batch, 1, 1, n,
                k * ਆਕਾਰ(ਦਸ਼ਮਲਵ), m * ਆਕਾਰ(ਦਸ਼ਮਲਵ)
            );
        }
    }
    ggml_vk_ctx_ਸਿਰਾ(subctx);

    ਸਵੈਚਾਲਿਤ begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ਸੱਚ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_ਕਤਾਰ_command_pools_cleanup(ctx->device);

    ਸਵੈਚਾਲਿਤ end = std::chrono::high_resolution_clock::now();

    ਦੁੱਗਣਾ ਸਮਾਂ_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_ਬਫਰ_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->ਅੰਕੜੇ = qx;
    src1_ggml->ਅੰਕੜੇ = y;
    tensor_ggml->ਅੰਕੜੇ = d_chk;

    ggml_cgraph * cgraph = ggml_new_ਗ੍ਰਾਫ(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_ਗ੍ਰਾਫ_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ਦੁੱਗਣਾ avg_err = 0.0;
    ਪੂਰਨ ਪਹਿਲਾ_err_n = -1;
    ਪੂਰਨ ਪਹਿਲਾ_err_m = -1;
    ਪੂਰਨ ਪਹਿਲਾ_err_b = -1;

    ਲਈ (size_t i = 0; i < m*n*batch; i++) {
        ਦੁੱਗਣਾ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ਜੇ ((err > 0.05f || std::isnan(err)) && ਪਹਿਲਾ_err_n == -1) {
            ਪਹਿਲਾ_err_b = i / (m * n);
            ਪਹਿਲਾ_err_n = (i % (m * n)) / m;
            ਪਹਿਲਾ_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ਦੁੱਗਣਾ tflops = 2.0*m*n*k*batch*ਅੰਕ_it / (ਸਮਾਂ_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    ਜੇ (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << ਵੰਡੋ_k << " matmul " << ਸਮਾਂ_ms / ਅੰਕ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::ਸਤਰ_ਅੰਤ;

    ਜੇ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << ਪਹਿਲਾ_err_m << " n = " << ਪਹਿਲਾ_err_n << " b = " << ਪਹਿਲਾ_err_b << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "Actual result: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(d, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);
        std::cerr << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "Expected result: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(d_chk, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

        std::cerr << "src0: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(x, GGML_TYPE_F32, k, m, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);
        std::cerr << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "src1: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(y, GGML_TYPE_F32, k, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

        ਜੇ (ਵੰਡੋ_k > 1) {
            ਦਸ਼ਮਲਵ * ਵੰਡੋ_k_buf = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k);
            ggml_vk_ਬਫਰ_read(ctx->prealloc_ਵੰਡੋ_k, 0, ਵੰਡੋ_k_buf, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * d_ne * ਵੰਡੋ_k);

            std::cerr << "d_buf0: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            std::cerr << "d_buf1: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf + d_ne, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            std::cerr << "d_buf2: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            std::cerr << "d_buf3: " << std::ਸਤਰ_ਅੰਤ << std::ਸਤਰ_ਅੰਤ;
            ggml_vk_print_ਮੈਟ੍ਰਿਕਸ_area(ਵੰਡੋ_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, ਪਹਿਲਾ_err_m, ਪਹਿਲਾ_err_n, ਪਹਿਲਾ_err_b);

            ਮੁਕਤ(ਵੰਡੋ_k_buf);
        }
    }

    ggml_vk_destroy_ਬਫਰ(qx_buf);
    ggml_vk_destroy_ਬਫਰ(y_buf);
    ggml_vk_destroy_ਬਫਰ(qy_buf);
    ggml_vk_destroy_ਬਫਰ(d_buf);

    ਮੁਕਤ(x);
    ਮੁਕਤ(qx);
    ਮੁਕਤ(y);
    ਮੁਕਤ(d);
    ਮੁਕਤ(d_chk);
}
#ਜੇ_ਅੰਤ

int64_t ggml_vk_get_op_batch_ਮਾਪ(ਅਚਲ ggml_tensor * op) {
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ 0;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ਵਾਪਸ op->ne[1];
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
        ਹਾਲਤ GGML_OP_ROPE:
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
            ਵਾਪਸ op->ne[2];
        ਮੂਲ:
            ਵਾਪਸ ggml_nrows(op);
    }
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_CHECK_RESULTS
ਸਥਿਰ ਖਾਲੀ ggml_vk_print_ਗ੍ਰਾਫ_origin(ਅਚਲ ggml_tensor * tensor, std::vector<ਅਚਲ ggml_tensor *>& ਪੂਰਾ, ਪੂਰਨ level = 0) {
    ਜੇ (std::find(ਪੂਰਾ.begin(), ਪੂਰਾ.end(), tensor) != ਪੂਰਾ.end() || level > 10) {
        ਵਾਪਸ;
    }
    ਲਈ (ਪੂਰਨ j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != ਖਾਲੀ_ਸੂਚਕ) << std::ਸਤਰ_ਅੰਤ;

    ਪੂਰਾ.push_back(tensor);

    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (tensor->src[i] != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_vk_print_ਗ੍ਰਾਫ_origin(tensor->src[i], ਪੂਰਾ, level + 1);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_print_tensor_area(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, ਪੂਰਨ i0, ਪੂਰਨ i1, ਪੂਰਨ i2, ਪੂਰਨ i3) {
    ਜੇ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        ਵਾਪਸ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    ਲਈ (ਪੂਰਨ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ਲਈ (ਪੂਰਨ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ਲਈ (ਪੂਰਨ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ਜੇ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                ਦਸ਼ਮਲਵ val;
                ਜੇ (tensor->type == GGML_TYPE_F32) {
                    val = *(ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ਨਹੀਂਤਾਂ ਜੇ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_ਮੰਜ਼ਿਲ_fp32(*(ਅਚਲ ggml_fp16_t *) ((ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } ਨਹੀਂਤਾਂ ਜੇ (tensor->type == GGML_TYPE_I32) {
                    val = *(ਅਚਲ int32_t *) ((ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ਨਹੀਂਤਾਂ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_vk_print_tensor(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ਅੱਖਰ * name) {
    ਖਾਲੀ * tensor_ਅੰਕੜੇ = tensor->ਅੰਕੜੇ;

    ਅਚਲ ਤਾਰਕਿਕ is_gpu = tensor->ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ && ggml_backend_ਬਫਰ_is_vk(tensor->ਬਫਰ);

    ਜੇ (is_gpu) {
        ਅਚਲ size_t tensor_ਮਾਪ = ggml_nbytes(tensor);
        tensor_ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(tensor_ਮਾਪ);

        ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)tensor->ਬਫਰ->context;

        vk_ਬਫਰ ਬਫਰ_gpu = buf_ctx->dev_ਬਫਰ;
        ggml_vk_ਬਫਰ_read(ਬਫਰ_gpu, vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs, tensor_ਅੰਕੜੇ, tensor_ਮਾਪ);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::ਸਤਰ_ਅੰਤ;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::ਸਤਰ_ਅੰਤ;
    ਜੇ (tensor->src[0] != ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::ਸਤਰ_ਅੰਤ;
    }
    ਜੇ (tensor->src[1] != ਖਾਲੀ_ਸੂਚਕ) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::ਸਤਰ_ਅੰਤ;
    }
    std::cerr << std::ਸਤਰ_ਅੰਤ << "Result:" << std::ਸਤਰ_ਅੰਤ;
    ggml_vk_print_tensor_area(tensor, tensor_ਅੰਕੜੇ, 5, 5, 0, 0);
    std::cerr << std::ਸਤਰ_ਅੰਤ;
    std::vector<ਅਚਲ ggml_tensor *> ਪੂਰਾ;
    ggml_vk_print_ਗ੍ਰਾਫ_origin(tensor, ਪੂਰਾ);

    ਜੇ (is_gpu) {
        ਮੁਕਤ(tensor_ਅੰਕੜੇ);
    }
}

ਖਾਲੀ * comp_ਨਤੀਜਾ;
size_t comp_ਮਾਪ;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
ਖਾਲੀ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ਪੂਰਨ tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->ਅੰਕ_additional_fused_ops];
    ਜੇ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }

    check_counter++;
    ਜੇ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        ਵਾਪਸ;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    ਬਣਤਰ ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਝੂਠ,
    };

    ਬਣਤਰ ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<ਬਣਤਰ ggml_tensor *, GGML_MAX_SRC> src_clone = {ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ};
    ਅਚਲ ਅੱਖਰ * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<ਖਾਲੀ *> cloned_mallocs;

    ਬਣਤਰ ggml_tensor * tensor_clone = ਖਾਲੀ_ਸੂਚਕ;

    ਲਈ (ਪੂਰਨ f = 0; f < ctx->ਅੰਕ_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            ਜੇ (srci == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            // If a src tensor has been cloned, use that one
            ਸਵੈਚਾਲਿਤ it = cloned_tensors.find(srci);
            ਜੇ (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                ਜਾਰੀ;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            size_t srci_ਮਾਪ = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            ਖਾਲੀ *src_ਬਫਰ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(srci_ਮਾਪ);
            cloned_mallocs.push_back(src_ਬਫਰ);

            srci_clone->ਅੰਕੜੇ = src_ਬਫਰ;
            ਜੇ (ggml_backend_ਬਫਰ_is_host(srci->ਬਫਰ)) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(srci_clone->ਅੰਕੜੇ, srci->ਅੰਕੜੇ, srci_ਮਾਪ);
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(srci_clone->nb, srci->nb, ਆਕਾਰ(size_t) * GGML_MAX_DIMS);
            } ਨਹੀਂਤਾਂ ਜੇ (ggml_backend_ਬਫਰ_is_vk(srci->ਬਫਰ)) {
                ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)srci->ਬਫਰ->context;
                vk_ਬਫਰ& ਬਫਰ_gpu = buf_ctx->dev_ਬਫਰ;
                uint64_t ਵਿੱਥ = vk_tensor_ਵਿੱਥ(srci) + srci->view_offs;
                ਜੇ (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    ਲਈ (ਪੂਰਨ i3 = 0; i3 < srci->ne[3]; i3++) {
                        ਲਈ (ਪੂਰਨ i2 = 0; i2 < srci->ne[2]; i2++) {
                            ਅਚਲ ਪੂਰਨ idx = i3*srci->ne[2] + i2;
                            ggml_vk_ਬਫਰ_read(ਬਫਰ_gpu, ਵਿੱਥ + idx * srci->nb[2], ((ਅੱਖਰ *)srci_clone->ਅੰਕੜੇ + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    ਲਈ (ਪੂਰਨ i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } ਨਹੀਂਤਾਂ {
                    ਜੇ (ਵਿੱਥ + srci_ਮਾਪ >= ਬਫਰ_gpu->size) {
                        srci_ਮਾਪ = ਬਫਰ_gpu->size - ਵਿੱਥ;
                    }
                    ggml_vk_ਬਫਰ_read(ਬਫਰ_gpu, ਵਿੱਥ, srci_clone->ਅੰਕੜੇ, srci_ਮਾਪ);
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(srci_clone->nb, srci->nb, ਆਕਾਰ(size_t) * GGML_MAX_DIMS);
                }
            } ਨਹੀਂਤਾਂ {
                GGML_ABORT("fatal error");
            }

            ਜੇ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        ਜੇ (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ਅਚਲ ਦਸ਼ਮਲਵ * params = (ਅਚਲ ਦਸ਼ਮਲਵ *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            ਜੇ (src_clone[4]) {
                ggml_flash_attn_ext_ਜੋੜੋ_sinks(tensor_clone, src_clone[4]);
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(ਪੂਰਨ *)tensor->op_params);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_ਬਹੁਲਕ) tensor->op_params[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SCALE) {
            ਅਚਲ ਦਸ਼ਮਲਵ * params = (ਅਚਲ ਦਸ਼ਮਲਵ *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ARANGE) {
            ਅਚਲ ਦਸ਼ਮਲਵ start = ggml_get_op_params_f32(tensor, 0);
            ਅਚਲ ਦਸ਼ਮਲਵ stop = ggml_get_op_params_f32(tensor, 1);
            ਅਚਲ ਦਸ਼ਮਲਵ ਕਦਮ = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, ਕਦਮ);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_FILL) {
            ਅਚਲ ਦਸ਼ਮਲਵ value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CLAMP) {
            ਅਚਲ ਦਸ਼ਮਲਵ * params = (ਅਚਲ ਦਸ਼ਮਲਵ *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_REPEAT_ਪਿੱਠ) {
            tensor_clone = ggml_repeat_ਪਿੱਠ(ggml_ctx, src_clone[0], tensor);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਜੋੜੋ) {
            tensor_clone = ggml_ਜੋੜੋ(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਸੈੱਟ) {
            tensor_clone = ggml_ਸੈੱਟ(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(ਦਸ਼ਮਲਵ *)tensor->op_params);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਸਮੂਹ_NORM) {
            ਅਚਲ ਦਸ਼ਮਲਵ * float_params = (ਅਚਲ ਦਸ਼ਮਲਵ *)tensor->op_params;
            tensor_clone = ggml_ਸਮੂਹ_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(ਦਸ਼ਮਲਵ *)tensor->op_params);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_RMS_NORM_ਪਿੱਠ) {
            ਅਚਲ ਦਸ਼ਮਲਵ eps = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_ਪਿੱਠ(ggml_ctx, src_clone[0], src_clone[1], eps);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SILU_ਪਿੱਠ) {
            tensor_clone = ggml_silu_ਪਿੱਠ(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_L2_NORM) {
            ਅਚਲ ਦਸ਼ਮਲਵ eps = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SOFT_MAX) {
            ਜੇ (tensor->src[1] != ਖਾਲੀ_ਸੂਚਕ) {
                ਅਚਲ ਦਸ਼ਮਲਵ * params = (ਅਚਲ ਦਸ਼ਮਲਵ *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } ਨਹੀਂਤਾਂ {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SOFT_MAX_ਪਿੱਠ) {
            tensor_clone = ggml_soft_max_ext_ਪਿੱਠ(ggml_ctx, src_clone[0], src_clone[1], ((ਦਸ਼ਮਲਵ *)tensor->op_params)[0], ((ਦਸ਼ਮਲਵ *)tensor->op_params)[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_DIAG_ਨਕਾਬ_INF) {
            tensor_clone = ggml_diag_ਨਕਾਬ_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_ਪਿੱਠ) {
            ਅਚਲ ਪੂਰਨ n_dims      = ((int32_t *) tensor->op_params)[1];
            ਅਚਲ ਪੂਰਨ mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            ਅਚਲ ਪੂਰਨ n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            ਅਚਲ ਦਸ਼ਮਲਵ freq_ਅਧਾਰ       = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[5];
            ਅਚਲ ਦਸ਼ਮਲਵ freq_scale      = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[6];
            ਅਚਲ ਦਸ਼ਮਲਵ ext_ਗੁਣਕ      = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[7];
            ਅਚਲ ਦਸ਼ਮਲਵ attn_ਗੁਣਕ     = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[8];
            ਅਚਲ ਦਸ਼ਮਲਵ beta_fast       = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[9];
            ਅਚਲ ਦਸ਼ਮਲਵ beta_slow       = ((ਦਸ਼ਮਲਵ *) tensor->op_params)[10];
            ਜੇ (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                ਜੇ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
                } ਨਹੀਂਤਾਂ {
                    tensor_clone = ggml_rope_multi_ਪਿੱਠ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
                } ਨਹੀਂਤਾਂ {
                    tensor_clone = ggml_rope_ext_ਪਿੱਠ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow);
                }
            }
            ਅਚਲ ਪੂਰਨ n_offs = ((int32_t *) tensor->op_params)[15];
            ਜੇ (n_offs != 0) {
                tensor_clone = ggml_rope_ਸੈੱਟ_ਵਿੱਥ(tensor_clone, n_offs);
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_UNARY) {
            ਚੋਣ (ggml_get_unary_op(tensor)) {
            ਹਾਲਤ GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                tensor_clone = ggml_gelu_ਤੇਜ਼(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_ਸੈੱਟ_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_ਸੈੱਟ_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_ਸੈੱਟ_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_ਸੈੱਟ_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                tensor_clone = ggml_ਕਦਮ(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                tensor_clone = ggml_ਗੇੜਾ(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                ਰੋਕੋ;
            ਮੂਲ:
                std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::ਸਤਰ_ਅੰਤ;
                GGML_ABORT("fatal error");
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_GLU) {
            ਜੇ (src_clone[1] == ਖਾਲੀ_ਸੂਚਕ) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } ਨਹੀਂਤਾਂ {
                tensor_clone = ggml_glu_ਵੰਡੋ(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_ਸੈੱਟ_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_ਸੈੱਟ_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            ਜੇ (tensor->src[1] == ਖਾਲੀ_ਸੂਚਕ) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } ਨਹੀਂਤਾਂ {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_ਕ੍ਰਮ_ਬਦਲੋ(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_GET_ਪੰਕਤੀਆਂ) {
            tensor_clone = ggml_get_ਪੰਕਤੀਆਂ(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_ਛਾਂਟੋ_ਕ੍ਰਮ) *(ਪੂਰਨ *)tensor->op_params);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਸਿਖਰ_K) {
            tensor_clone = ggml_ਸਿਖਰ_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਜੋੜ) {
            tensor_clone = ggml_ਜੋੜ(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ) {
            tensor_clone = ggml_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_DSV4_HC_PRE) {
            ਜੇ (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } ਨਹੀਂਤਾਂ {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਔਸਤ) {
            tensor_clone = ggml_ਔਸਤ(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ) {
            tensor_clone = ggml_cross_entropy_loss_ਪਿੱਠ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਗਿਣਤੀ_EQUAL) {
            tensor_clone = ggml_ਗਿਣਤੀ_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], ਸੱਚ, ਸੱਚ, ਝੂਠ);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_IM2_ਖਾਨਾ) {
            ਅਚਲ int32_t s0 = tensor->op_params[0];
            ਅਚਲ int32_t s1 = tensor->op_params[1];
            ਅਚਲ int32_t p0 = tensor->op_params[2];
            ਅਚਲ int32_t p1 = tensor->op_params[3];
            ਅਚਲ int32_t d0 = tensor->op_params[4];
            ਅਚਲ int32_t d1 = tensor->op_params[5];

            ਅਚਲ ਤਾਰਕਿਕ is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_IM2_ਖਾਨਾ_3_D) {
            ਅਚਲ int32_t s0 = tensor->op_params[0];
            ਅਚਲ int32_t s1 = tensor->op_params[1];
            ਅਚਲ int32_t s2 = tensor->op_params[2];
            ਅਚਲ int32_t p0 = tensor->op_params[3];
            ਅਚਲ int32_t p1 = tensor->op_params[4];
            ਅਚਲ int32_t p2 = tensor->op_params[5];
            ਅਚਲ int32_t d0 = tensor->op_params[6];
            ਅਚਲ int32_t d1 = tensor->op_params[7];
            ਅਚਲ int32_t d2 = tensor->op_params[8];
            ਅਚਲ int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            ਅਚਲ int32_t dim = tensor->op_params[0];
            ਅਚਲ int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            ਅਚਲ int32_t s0 = tensor->op_params[0];
            ਅਚਲ int32_t p0 = tensor->op_params[1];
            ਅਚਲ int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_COL2IM_1D) {
            ਅਚਲ int32_t stride = tensor->op_params[0];
            ਅਚਲ int32_t oc     = tensor->op_params[1];
            ਅਚਲ int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_POOL_1D) {
            ਗਿਣਤੀ ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ਅਚਲ int32_t k0 = tensor->op_params[1];
            ਅਚਲ int32_t s0 = tensor->op_params[2];
            ਅਚਲ int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_POOL_2D) {
            ਗਿਣਤੀ ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ਅਚਲ int32_t k0 = tensor->op_params[1];
            ਅਚਲ int32_t k1 = tensor->op_params[2];
            ਅਚਲ int32_t s0 = tensor->op_params[3];
            ਅਚਲ int32_t s1 = tensor->op_params[4];
            ਅਚਲ int32_t p0 = tensor->op_params[5];
            ਅਚਲ int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CONV_2D) {
            ਅਚਲ int32_t s0 = tensor->op_params[0];
            ਅਚਲ int32_t s1 = tensor->op_params[1];
            ਅਚਲ int32_t p0 = tensor->op_params[2];
            ਅਚਲ int32_t p1 = tensor->op_params[3];
            ਅਚਲ int32_t d0 = tensor->op_params[4];
            ਅਚਲ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CONV_3D) {
            ਅਚਲ int32_t s0 = tensor->op_params[0];
            ਅਚਲ int32_t s1 = tensor->op_params[1];
            ਅਚਲ int32_t s2 = tensor->op_params[2];
            ਅਚਲ int32_t p0 = tensor->op_params[3];
            ਅਚਲ int32_t p1 = tensor->op_params[4];
            ਅਚਲ int32_t p2 = tensor->op_params[5];
            ਅਚਲ int32_t d0 = tensor->op_params[6];
            ਅਚਲ int32_t d1 = tensor->op_params[7];
            ਅਚਲ int32_t d2 = tensor->op_params[8];
            ਅਚਲ int32_t IC = tensor->op_params[9];
            ਅਚਲ int32_t N  = tensor->op_params[10];
            ਅਚਲ int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_ਸਿੱਧਾ(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CONV_2D_DW) {
            ਅਚਲ int32_t s0 = tensor->op_params[0];
            ਅਚਲ int32_t s1 = tensor->op_params[1];
            ਅਚਲ int32_t p0 = tensor->op_params[2];
            ਅਚਲ int32_t p1 = tensor->op_params[3];
            ਅਚਲ int32_t d0 = tensor->op_params[4];
            ਅਚਲ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_ਸਿੱਧਾ(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            ਅਚਲ int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_LEAKY_RELU) {
            ਅਚਲ ਦਸ਼ਮਲਵ * op_params = (ਅਚਲ ਦਸ਼ਮਲਵ *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], ਝੂਠ);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_GATED_ਰੇਖਿਕ_ATTN) {
            ਅਚਲ ਦਸ਼ਮਲਵ * op_params = (ਅਚਲ ਦਸ਼ਮਲਵ *)tensor->op_params;
            tensor_clone = ggml_gated_ਰੇਖਿਕ_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_OPT_ਕਦਮ_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_ਕਦਮ_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_OPT_ਕਦਮ_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_ਕਦਮ_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ਜੋੜੋ_ID) {
            tensor_clone = ggml_ਜੋੜੋ_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SSM_ਜਾਂਚ) {
            ਅਚਲ int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_ਜਾਂਚ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } ਨਹੀਂਤਾਂ ਜੇ (tensor->op == GGML_OP_ROLL) {
            ਅਚਲ int32_t s0 = tensor->op_params[0];
            ਅਚਲ int32_t s1 = tensor->op_params[1];
            ਅਚਲ int32_t s2 = tensor->op_params[2];
            ਅਚਲ int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        ਨਹੀਂਤਾਂ {
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::ਸਤਰ_ਅੰਤ;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_ਗ੍ਰਾਫ(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_ਗ੍ਰਾਫ_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    ਜੇ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_ਮਾਪ = ggml_nbytes(tensor_clone);

    comp_ਨਤੀਜਾ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(comp_ਮਾਪ);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(comp_ਨਤੀਜਾ, tensor_clone->ਅੰਕੜੇ, comp_ਮਾਪ);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(comp_nb, tensor_clone->nb, ਆਕਾਰ(size_t) * GGML_MAX_DIMS);

    ਲਈ (ਸਵੈਚਾਲਿਤ m : cloned_mallocs) {
        ਮੁਕਤ(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

ਖਾਲੀ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ਪੂਰਨ tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->ਅੰਕ_additional_fused_ops];
    ਜੇ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }

    ਜੇ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        ਵਾਪਸ;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    ਖਾਲੀ * tensor_ਅੰਕੜੇ = tensor->ਅੰਕੜੇ;

    ਜੇ (ggml_backend_ਬਫਰ_is_vk(tensor->ਬਫਰ)) {
        size_t tensor_ਮਾਪ = ggml_nbytes(tensor);
        tensor_ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(tensor_ਮਾਪ);

        ggml_backend_vk_ਬਫਰ_context * buf_ctx = (ggml_backend_vk_ਬਫਰ_context *)tensor->ਬਫਰ->context;

        vk_ਬਫਰ& ਬਫਰ_gpu = buf_ctx->dev_ਬਫਰ;
        uint64_t ਵਿੱਥ = vk_tensor_ਵਿੱਥ(tensor) + tensor->view_offs;
        ਜੇ (ਵਿੱਥ + tensor_ਮਾਪ >= ਬਫਰ_gpu->size) {
            tensor_ਮਾਪ = ਬਫਰ_gpu->size - ਵਿੱਥ;
        }

        ggml_vk_ਬਫਰ_read(ਬਫਰ_gpu, ਵਿੱਥ, tensor_ਅੰਕੜੇ, tensor_ਮਾਪ);
    }

    ਦਸ਼ਮਲਵ ਪਹਿਲਾ_error_ਨਤੀਜਾ = -1.0f;
    ਦਸ਼ਮਲਵ ਪਹਿਲਾ_error_correct = -1.0f;
    std::array<ਪੂਰਨ, 4> ਪਹਿਲਾ_error = { -1, -1, -1, -1 };
    ਦੁੱਗਣਾ avg_err = 0.0;
    size_t counter = 0;

    ਲਈ (ਪੂਰਨ i3 = 0; i3 < tensor->ne[3]; i3++) {
        ਲਈ (ਪੂਰਨ i2 = 0; i2 < tensor->ne[2]; i2++) {
            ਲਈ (ਪੂਰਨ i1 = 0; i1 < tensor->ne[1]; i1++) {
                ਲਈ (ਪੂਰਨ i0 = 0; i0 < tensor->ne[0]; i0++) {
                    ਅਚਲ ਤਾਰਕਿਕ ਬਫਰ_ਮਾਪ_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_ਮਾਪ;
                    ਦਸ਼ਮਲਵ correct = 0.0f;
                    ਦਸ਼ਮਲਵ ਨਤੀਜਾ = 0.0f;

                    ਜੇ (ਬਫਰ_ਮਾਪ_fit) {
                        ਜੇ (tensor->type == GGML_TYPE_F32) {
                            correct = *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) comp_ਨਤੀਜਾ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ਨਤੀਜਾ  = *(ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) tensor_ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ਨਹੀਂਤਾਂ ਜੇ (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_ਮੰਜ਼ਿਲ_fp32(*(ggml_fp16_t *) ((ਅੱਖਰ *) comp_ਨਤੀਜਾ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ਨਤੀਜਾ  = ggml_fp16_ਮੰਜ਼ਿਲ_fp32(*(ggml_fp16_t *) ((ਅੱਖਰ *) tensor_ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } ਨਹੀਂਤਾਂ ਜੇ (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_ਮੰਜ਼ਿਲ_fp32(*(ggml_bf16_t *) ((ਅੱਖਰ *) comp_ਨਤੀਜਾ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ਨਤੀਜਾ  = ggml_bf16_ਮੰਜ਼ਿਲ_fp32(*(ggml_bf16_t *) ((ਅੱਖਰ *) tensor_ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } ਨਹੀਂਤਾਂ ਜੇ (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((ਅੱਖਰ *) comp_ਨਤੀਜਾ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ਨਤੀਜਾ  = *(int32_t *) ((ਅੱਖਰ *) tensor_ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ਨਹੀਂਤਾਂ ਜੇ (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((ਅੱਖਰ *) comp_ਨਤੀਜਾ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ਨਤੀਜਾ  = *(int64_t *) ((ਅੱਖਰ *) tensor_ਅੰਕੜੇ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ਨਹੀਂਤਾਂ {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::ਸਤਰ_ਅੰਤ;
                        }
                    } ਨਹੀਂਤਾਂ {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::ਸਤਰ_ਅੰਤ;
                        GGML_ABORT("fatal error");
                    }

                    ਜੇ ((std::isnan(correct) != std::isnan(ਨਤੀਜਾ)) || (std::isinf(correct) != std::isinf(ਨਤੀਜਾ)) || !ਬਫਰ_ਮਾਪ_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << ਨਤੀਜਾ << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::ਸਤਰ_ਅੰਤ;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::ਸਤਰ_ਅੰਤ;
                        ਜੇ (src0 != ਖਾਲੀ_ਸੂਚਕ) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::ਸਤਰ_ਅੰਤ;
                        }
                        ਜੇ (src1 != ਖਾਲੀ_ਸੂਚਕ) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::ਸਤਰ_ਅੰਤ;
                        }
                        ਜੇ (src2 != ਖਾਲੀ_ਸੂਚਕ) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::ਸਤਰ_ਅੰਤ;
                        }
                        ਜੇ (src3 != ਖਾਲੀ_ਸੂਚਕ) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::ਸਤਰ_ਅੰਤ;
                        }
                        std::cerr << "First error: result=" << ਪਹਿਲਾ_error_ਨਤੀਜਾ << " correct=" << ਪਹਿਲਾ_error_correct  << " i3=" << ਪਹਿਲਾ_error[3] << " i2=" << ਪਹਿਲਾ_error[2] << " i1=" << ਪਹਿਲਾ_error[1] << " i0=" << ਪਹਿਲਾ_error[0] << std::ਸਤਰ_ਅੰਤ;
                        std::cerr << std::ਸਤਰ_ਅੰਤ << "Result:" << std::ਸਤਰ_ਅੰਤ;
                        ggml_vk_print_tensor_area(tensor, tensor_ਅੰਕੜੇ, i0, i1, i2, i3);
                        std::cerr << std::ਸਤਰ_ਅੰਤ << "Correct:" << std::ਸਤਰ_ਅੰਤ;
                        ggml_vk_print_tensor_area(tensor, comp_ਨਤੀਜਾ, i0, i1, i2, i3);
                        std::cerr << std::ਸਤਰ_ਅੰਤ;
                        std::vector<ਅਚਲ ggml_tensor *> ਪੂਰਾ;
                        ggml_vk_print_ਗ੍ਰਾਫ_origin(tensor, ਪੂਰਾ);
                        GGML_ABORT("fatal error");
                    }
                    ਅਚਲ ਦੁੱਗਣਾ denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    ਜੇ (ਪਹਿਲਾ_error[0] == -1 && std::fabs(correct - ਨਤੀਜਾ) / denom > 0.5) {
                        ਪਹਿਲਾ_error[0] = i0;
                        ਪਹਿਲਾ_error[1] = i1;
                        ਪਹਿਲਾ_error[2] = i2;
                        ਪਹਿਲਾ_error[3] = i3;
                        ਪਹਿਲਾ_error_ਨਤੀਜਾ = ਨਤੀਜਾ;
                        ਪਹਿਲਾ_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    ਜੇ (!std::isinf(correct) && !std::isinf(ਨਤੀਜਾ) && !std::isnan(correct) && !std::isnan(ਨਤੀਜਾ)) {
                        avg_err += std::fabs(correct - ਨਤੀਜਾ) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    ਜੇ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::ਸਤਰ_ਅੰਤ;
        ਜੇ (src0 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        ਜੇ (src1 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        ਜੇ (src2 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        ਜੇ (src3 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        std::cerr << "First error: result=" << ਪਹਿਲਾ_error_ਨਤੀਜਾ << " correct=" << ਪਹਿਲਾ_error_correct  << " i3=" << ਪਹਿਲਾ_error[3] << " i2=" << ਪਹਿਲਾ_error[2] << " i1=" << ਪਹਿਲਾ_error[1] << " i0=" << ਪਹਿਲਾ_error[0] << std::ਸਤਰ_ਅੰਤ;
        std::cerr << std::ਸਤਰ_ਅੰਤ << "Result:" << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_tensor_area(tensor, tensor_ਅੰਕੜੇ, 5, 5, 0, 0);
        std::cerr << std::ਸਤਰ_ਅੰਤ << "Correct:" << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_tensor_area(tensor, comp_ਨਤੀਜਾ, 5, 5, 0, 0);
        std::cerr << std::ਸਤਰ_ਅੰਤ;
        std::vector<ਅਚਲ ggml_tensor *> ਪੂਰਾ;
        ggml_vk_print_ਗ੍ਰਾਫ_origin(tensor, ਪੂਰਾ);
    }

    ਜੇ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::ਸਤਰ_ਅੰਤ;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::ਸਤਰ_ਅੰਤ;
        ਜੇ (src0 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        ਜੇ (src1 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        ਜੇ (src2 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        ਜੇ (src3 != ਖਾਲੀ_ਸੂਚਕ) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::ਸਤਰ_ਅੰਤ;
        }
        std::cerr << "First error: result=" << ਪਹਿਲਾ_error_ਨਤੀਜਾ << " correct=" << ਪਹਿਲਾ_error_correct  << " i3=" << ਪਹਿਲਾ_error[3] << " i2=" << ਪਹਿਲਾ_error[2] << " i1=" << ਪਹਿਲਾ_error[1] << " i0=" << ਪਹਿਲਾ_error[0] << std::ਸਤਰ_ਅੰਤ;
        std::cerr << std::ਸਤਰ_ਅੰਤ << "Result:" << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_tensor_area(tensor, tensor_ਅੰਕੜੇ, ਪਹਿਲਾ_error[0], ਪਹਿਲਾ_error[1], ਪਹਿਲਾ_error[2], ਪਹਿਲਾ_error[3]);
        std::cerr << std::ਸਤਰ_ਅੰਤ << "Correct:" << std::ਸਤਰ_ਅੰਤ;
        ggml_vk_print_tensor_area(tensor, comp_ਨਤੀਜਾ, ਪਹਿਲਾ_error[0], ਪਹਿਲਾ_error[1], ਪਹਿਲਾ_error[2], ਪਹਿਲਾ_error[3]);
        std::cerr << std::ਸਤਰ_ਅੰਤ;
        std::vector<ਅਚਲ ggml_tensor *> ਪੂਰਾ;
        ggml_vk_print_ਗ੍ਰਾਫ_origin(tensor, ਪੂਰਾ);
        GGML_ABORT("fatal error");
    } ਨਹੀਂਤਾਂ {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::ਸਤਰ_ਅੰਤ;
    }

    ਮੁਕਤ(comp_ਨਤੀਜਾ);
    comp_ਨਤੀਜਾ = ਖਾਲੀ_ਸੂਚਕ;
    comp_ਮਾਪ = 0;

    ਜੇ (ggml_backend_ਬਫਰ_is_vk(tensor->ਬਫਰ)) {
        ਮੁਕਤ(tensor_ਅੰਕੜੇ);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#ਜੇ_ਅੰਤ

