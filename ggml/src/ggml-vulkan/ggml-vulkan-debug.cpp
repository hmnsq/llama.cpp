#ಅನ್ತರ್ಭಾವ "ggml-vulkan-common.h"

ತಾರ್ಕಿಕ vk_ಸ್ಮರಣೆ_logger_enabled = ಸುಳ್ಳು;

ತಾರ್ಕಿಕ vk_perf_logger_enabled = ಸುಳ್ಳು;

ತಾರ್ಕಿಕ vk_perf_logger_concurrent = ಸುಳ್ಳು;

ತಾರ್ಕಿಕ vk_enable_sync_logger = ಸುಳ್ಳು;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

ಶೂನ್ಯ vk_ಸ್ಮರಣೆ_logger::log_ಹಂಚಿಕೆ(vk_ಬಫರ್_ref buf_ref, size_t size) {
    ಆದರೆ (!vk_ಸ್ಮರಣೆ_logger_enabled) {
        ಹಿಂತಿರುಗಿ;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_ಬಫರ್ buf = buf_ref.ಬೀಗ();
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ device = ತಾರ್ಕಿಕ(buf->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal);
    ಸ್ಥಿರಾಂಕ std::string type = device ? "device" : "host";
    allocations[buf->ಬಫರ್] = size;
    ಒಟ್ಟು_device += device ? size : 0;
    ಒಟ್ಟು_host += device ? 0 : size;
    VK_LOG_ಸ್ಮರಣೆ(buf->device->name << ": +" << format_ಗಾತ್ರ(size) << " " << type << " at " << buf->ಬಫರ್ << ". Total device: " << format_ಗಾತ್ರ(ಒಟ್ಟು_device) << ", total host: " << format_ಗಾತ್ರ(ಒಟ್ಟು_host));
}

ಶೂನ್ಯ vk_ಸ್ಮರಣೆ_logger::log_deallocation(vk_ಬಫರ್_ref buf_ref) {
    ಆದರೆ (buf_ref.expired() || buf_ref.ಬೀಗ()->size == 0 || !vk_ಸ್ಮರಣೆ_logger_enabled) {
        ಹಿಂತಿರುಗಿ;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_ಬಫರ್ buf = buf_ref.ಬೀಗ();
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ device = ತಾರ್ಕಿಕ(buf->ಸ್ಮರಣೆ_property_flags & vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal);
    std::string type = device ? "device" : "host";
    ಸ್ವಯಂಚಾಲಿತ it = allocations.find(buf->ಬಫರ್);
    ಆದರೆ (it != allocations.end()) {
        ಒಟ್ಟು_device -= device ? it->second : 0;
        ಒಟ್ಟು_host -= device ? 0 : it->second;
        VK_LOG_ಸ್ಮರಣೆ(buf->device->name << ": -" << format_ಗಾತ್ರ(it->second) << " " << type << " at " << buf->ಬಫರ್ << ". Total device: " << format_ಗಾತ್ರ(ಒಟ್ಟು_device) << ", total host: " << format_ಗಾತ್ರ(ಒಟ್ಟು_host));
        allocations.erase(it);
    } ಇಲ್ಲದಿದ್ದರೆ {
        VK_LOG_ಸ್ಮರಣೆ("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->ಬಫರ್);
    }
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_ಸತತ_TESTS
ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_ಮಾತೃಕೆ_area(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, ggml_type type, ಪೂರ್ಣಾಂಕ ne0, ಪೂರ್ಣಾಂಕ ne1, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1, ಪೂರ್ಣಾಂಕ i2) {
    ಆದರೆ (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ಆದರೆ (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                ದಶಮಾಂಶ val;
                ಆದರೆ (type == GGML_TYPE_F32) {
                    val = *((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ದತ್ತಾಂಶ + i2*ne1*ne0 + idx1*ne0 + idx0);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == GGML_TYPE_F16) {
                    val = ggml_fp16_ಗಮ್ಯ_fp32(*((ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ದತ್ತಾಂಶ + i2*ne1*ne0 + idx1*ne0 + idx0));
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ಇಲ್ಲದಿದ್ದರೆ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು X_TYPE, ಬಗೆ_ಹೆಸರು Y_TYPE>
ಶೂನ್ಯ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_k, ಪೂರ್ಣಾಂಕ shader_ಗಾತ್ರ) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << ಅಂಕೆ_it << ", " << ವಿಭಜಿಸು_k << ", " << shader_ಗಾತ್ರ << ")");
    ಸ್ಥಿರಾಂಕ size_t x_ne = m * k * batch;
    ಸ್ಥಿರಾಂಕ size_t y_ne = k * n * batch;
    ಸ್ಥಿರಾಂಕ size_t d_ne = m * n * batch;

    ggml_type x_type = std::is_same<ದಶಮಾಂಶ, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<ದಶಮಾಂಶ, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_ಕೀಲಿ mm_test_ಕೀಲಿ{x_type, y_type, ಸುಳ್ಳು, ಸುಳ್ಳು};
    ಸ್ವಯಂಚಾಲಿತ mm_test_it = ctx->device->pipeline_matmul.find(mm_test_ಕೀಲಿ);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.ಖಾಲಿಗಳು());
    ಸ್ವಯಂಚಾಲಿತ& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_ಗಾತ್ರ >= 0 && shader_ಗಾತ್ರ < (ಪೂರ್ಣಾಂಕ)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_ಗಾತ್ರ);
    vk_pipeline p = mm_test_configs[shader_ಗಾತ್ರ].aligned ? mm_test_configs[shader_ಗಾತ್ರ].aligned : mm_test_configs[shader_ಗಾತ್ರ].unaligned;

    ಸ್ಥಿರಾಂಕ size_t kpad = ggml_vk_align_ಗಾತ್ರ(k, mm_test_configs[shader_ಗಾತ್ರ].align);

    ಆದರೆ (k != kpad) {
        p = mm_test_configs[shader_ಗಾತ್ರ].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_ಗಾತ್ರ);
    }

    ಆದರೆ (ವಿಭಜಿಸು_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ವಿಭಜಿಸು_k_reduce, ಅಂಕೆ_it);

        ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k == ಶೂನ್ಯಸೂಚಕ || ctx->prealloc_ವಿಭಜಿಸು_k->size < ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k) {
            // Resize buffer
            ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k != ಶೂನ್ಯಸೂಚಕ) {
                ggml_vk_destroy_ಬಫರ್(ctx->prealloc_ವಿಭಜಿಸು_k);
            }
            ctx->prealloc_ವಿಭಜಿಸು_k = ggml_vk_create_ಬಫರ್_check(ctx->device, ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_ಬಫರ್ d_X = ggml_vk_create_ಬಫರ್_check(ctx->device, ಗಾತ್ರ(X_TYPE) * x_ne, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    vk_ಬಫರ್ d_Y = ggml_vk_create_ಬಫರ್_check(ctx->device, ಗಾತ್ರ(Y_TYPE) * y_ne, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    vk_ಬಫರ್ d_D = ggml_vk_create_ಬಫರ್_check(ctx->device, ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(Y_TYPE) * y_ne);
    ದಶಮಾಂಶ* d = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne);

    ಗಾಗಿ (size_t i = 0; i < x_ne; i++) {
        ಆದರೆ (std::is_same<ದಶಮಾಂಶ, X_TYPE>()) {
            x[i] = (rand() / (ದಶಮಾಂಶ)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_ಗಮ್ಯ_fp16((rand() / (ದಶಮಾಂಶ)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("fatal error");
        }
    }
    ಗಾಗಿ (size_t i = 0; i < y_ne; i++) {
        ಆದರೆ (std::is_same<ದಶಮಾಂಶ, Y_TYPE>()) {
            y[i] = (rand() / (ದಶಮಾಂಶ)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_ಗಮ್ಯ_fp16((rand() / (ದಶಮಾಂಶ)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_ಬಫರ್_write(d_X, 0, x, ಗಾತ್ರ(X_TYPE) * k * m * batch);
    ggml_vk_ಬಫರ್_write(d_Y, 0, y, ಗಾತ್ರ(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_ವಿಭಜಿಸು_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            ವಿಭಜಿಸು_k, batch, batch, batch, 1, 1, n,
            k * ಗಾತ್ರ(Y_TYPE), m * ಗಾತ್ರ(ದಶಮಾಂಶ)
        );
    }
    ggml_vk_ctx_ಕೊನೆ(subctx);

    ಸ್ವಯಂಚಾಲಿತ begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ನಿಜ, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_ಸರತಿ_command_pools_cleanup(ctx->device);

    ಸ್ವಯಂಚಾಲಿತ end = std::chrono::high_resolution_clock::now();
    ದ್ವಿಗುಣ ಸಮಯ = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_ಬಫರ್_read(d_D, 0, d, ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne);

    ದಶಮಾಂಶ * d_chk = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    ಆದರೆ (std::is_same<ದಶಮಾಂಶ, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }
    ಆದರೆ (std::is_same<ದಶಮಾಂಶ, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->ದತ್ತಾಂಶ = x;
    src1_ggml->ದತ್ತಾಂಶ = y;
    tensor_ggml->ದತ್ತಾಂಶ = d_chk;

    ggml_cgraph * cgraph = ggml_new_ನಕ್ಷೆ(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_ನಕ್ಷೆ_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ದ್ವಿಗುಣ avg_err = 0.0;
    ಪೂರ್ಣಾಂಕ ಮೊದಲ_err_n = -1;
    ಪೂರ್ಣಾಂಕ ಮೊದಲ_err_m = -1;
    ಪೂರ್ಣಾಂಕ ಮೊದಲ_err_b = -1;

    ಗಾಗಿ (size_t i = 0; i < m*n*batch; i++) {
        ದ್ವಿಗುಣ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ಆದರೆ ((err > 0.05f || std::isnan(err)) && ಮೊದಲ_err_n == -1) {
            ಮೊದಲ_err_b = i / (m * n);
            ಮೊದಲ_err_n = (i % (m * n)) / m;
            ಮೊದಲ_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ದ್ವಿಗುಣ tflops = 2.0*m*n*k*batch*ಅಂಕೆ_it / (ಸಮಯ / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << ವಿಭಜಿಸು_k << " matmul " << ಸಮಯ / ಅಂಕೆ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::ಸಾಲು_ಅಂತ್ಯ;

    ಆದರೆ (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << ಮೊದಲ_err_m << " n = " << ಮೊದಲ_err_n << " b = " << ಮೊದಲ_err_b << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "Actual result: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_ಮಾತೃಕೆ_area(d, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);
        std::cerr << "Expected result: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_ಮಾತೃಕೆ_area(d_chk, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

        ಆದರೆ (ವಿಭಜಿಸು_k > 1) {
            ದಶಮಾಂಶ * ವಿಭಜಿಸು_k_buf = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k);
            ggml_vk_ಬಫರ್_read(ctx->prealloc_ವಿಭಜಿಸು_k, 0, ವಿಭಜಿಸು_k_buf, ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k);

            std::cerr << "d_buf0: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            std::cerr << "d_buf1: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf + d_ne, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            std::cerr << "d_buf2: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            std::cerr << "d_buf3: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            ಬಿಡುಗಡೆ(ವಿಭಜಿಸು_k_buf);
        }
    }

    ಬಿಡುಗಡೆ(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_ಬಫರ್(d_X);
    ggml_vk_destroy_ಬಫರ್(d_Y);
    ggml_vk_destroy_ಬಫರ್(d_D);

    ಬಿಡುಗಡೆ(x);
    ಬಿಡುಗಡೆ(y);
    ಬಿಡುಗಡೆ(d);
}

ಮಾದರಿ ಶೂನ್ಯ ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_k, ಪೂರ್ಣಾಂಕ shader_ಗಾತ್ರ);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor_area(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1, ಪೂರ್ಣಾಂಕ i2, ಪೂರ್ಣಾಂಕ i3) {
    ಆದರೆ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ಆದರೆ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                ದಶಮಾಂಶ val;
                ಆದರೆ (tensor->type == GGML_TYPE_F32) {
                    val = *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_ಗಮ್ಯ_fp32(*(ggml_fp16_t *) ((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ಇಲ್ಲದಿದ್ದರೆ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_quantize_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * from, ಶೂನ್ಯ * ಗಮ್ಯ, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, ಗಮ್ಯ, 0, 1, ne, ಶೂನ್ಯಸೂಚಕ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_dequantize_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * from, ದಶಮಾಂಶ * ಗಮ್ಯ, size_t ne, ggml_type quant) {
    ಆದರೆ (quant == GGML_TYPE_F32) {
        ಸ್ಮರಣೆ_ನಕಲು(ಗಮ್ಯ, from, ಗಾತ್ರ(ದಶಮಾಂಶ) * ne);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * tt = ggml_get_type_traits(quant);

    ggml_ಗಮ್ಯ_float_t dequant_fn = tt->ಗಮ್ಯ_float;

    dequant_fn(from, ಗಮ್ಯ, ne);
}

ಶೂನ್ಯ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    ಸ್ಥಿರಾಂಕ size_t x_sz = ಗಾತ್ರ(ದಶಮಾಂಶ) * ne;
    ಸ್ಥಿರಾಂಕ size_t x_sz_f16 = ಗಾತ್ರ(ggml_fp16_t) * ne;
    ಸ್ಥಿರಾಂಕ size_t qx_sz = ne * ggml_type_ಗಾತ್ರ(quant)/ggml_blck_ಗಾತ್ರ(quant);
    ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(x_sz);
    ಶೂನ್ಯ * qx = ಸ್ಮರಣೆ_ಕೊಡು(qx_sz);
    vk_ಬಫರ್ qx_buf = ggml_vk_create_ಬಫರ್_check(ctx->device, qx_sz, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    vk_ಬಫರ್ x_buf = ggml_vk_create_ಬಫರ್_check(ctx->device, x_sz_f16, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    ದಶಮಾಂಶ * x_ref = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) ಸ್ಮರಣೆ_ಕೊಡು(x_sz_f16);

    ಗಾಗಿ (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (ದಶಮಾಂಶ)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, quant);

    ggml_vk_quantize_ದತ್ತಾಂಶ(x, qx, ne, quant);
    ggml_vk_dequantize_ದತ್ತಾಂಶ(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_ಬಫರ್_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ಸ್ಥಿರಾಂಕ std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_ಕೊನೆ(subctx);

    ಸ್ವಯಂಚಾಲಿತ begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ನಿಜ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_ಸರತಿ_command_pools_cleanup(ctx->device);

    ಸ್ವಯಂಚಾಲಿತ end = std::chrono::high_resolution_clock::now();

    ದ್ವಿಗುಣ ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_ಬಫರ್_read(x_buf, 0, x_chk, x_sz_f16);

    ಪೂರ್ಣಾಂಕ ಮೊದಲ_err = -1;

    ದ್ವಿಗುಣ avg_err = 0.0;
    ಗಾಗಿ (size_t i = 0; i < ne; i++) {
        ದ್ವಿಗುಣ error = std::fabs(x_ref[i] - ggml_fp16_ಗಮ್ಯ_fp32(x_chk[i]));
        avg_err += error;

        ಆದರೆ (ಮೊದಲ_err < 0 && error > 0.05) {
            ಮೊದಲ_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::ಸಾಲು_ಅಂತ್ಯ;

    ಆದರೆ (avg_err > 0.1) {
        std::cerr << "first_error = " << ಮೊದಲ_err << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "Actual result: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = std::max(0, ಮೊದಲ_err - 5); i < std::min((ಪೂರ್ಣಾಂಕ)ne, ಮೊದಲ_err + 5); i++) {
            std::cerr << ggml_fp16_ಗಮ್ಯ_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Expected result: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = std::max(0, ಮೊದಲ_err - 5); i < std::min((ಪೂರ್ಣಾಂಕ)ne, ಮೊದಲ_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
    }

    ggml_vk_destroy_ಬಫರ್(x_buf);
    ggml_vk_destroy_ಬಫರ್(qx_buf);

    ಬಿಡುಗಡೆ(x);
    ಬಿಡುಗಡೆ(qx);
    ಬಿಡುಗಡೆ(x_ref);
    ಬಿಡುಗಡೆ(x_chk);
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

ಶೂನ್ಯ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, size_t ವಿಭಜಿಸು_k, size_t shader_ಗಾತ್ರ, ggml_type quant, ತಾರ್ಕಿಕ mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << ಅಂಕೆ_it << ", " << ವಿಭಜಿಸು_k << ", " << ggml_type_name(quant) << ")");
    ಸ್ಥಿರಾಂಕ size_t x_ne = m * k * batch;
    ಸ್ಥಿರಾಂಕ size_t y_ne = k * n * batch;
    ಸ್ಥಿರಾಂಕ size_t d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    ತಾರ್ಕಿಕ f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_ಕೀಲಿ dq_ಕೀಲಿ{quant, b_type, ಸುಳ್ಳು, f16acc};
    ಸ್ವಯಂಚಾಲಿತ dq_it = ctx->device->pipeline_matmul.find(dq_ಕೀಲಿ);
    ಆದರೆ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.ಖಾಲಿಗಳು()) {
        ಆದರೆ (f16acc) {
            dq_ಕೀಲಿ.f16acc = ಸುಳ್ಳು;
            dq_it = ctx->device->pipeline_matmul.find(dq_ಕೀಲಿ);
        }
    }
    ಆದರೆ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.ಖಾಲಿಗಳು()) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::ಸಾಲು_ಅಂತ್ಯ;
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ವಯಂಚಾಲಿತ& dq_configs = dq_it->second;
    ಆದರೆ (shader_ಗಾತ್ರ >= (ಪೂರ್ಣಾಂಕ)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_ಗಾತ್ರ << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::ಸಾಲು_ಅಂತ್ಯ;
        ಹಿಂತಿರುಗಿ;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_ಗಾತ್ರ);
    vk_pipeline p = dq_configs[shader_ಗಾತ್ರ].aligned ? dq_configs[shader_ಗಾತ್ರ].aligned : dq_configs[shader_ಗಾತ್ರ].unaligned;

    ಸ್ಥಿರಾಂಕ size_t kpad = mmq ? 0 : ggml_vk_align_ಗಾತ್ರ(k, dq_configs[shader_ಗಾತ್ರ].align);

    ಆದರೆ (mmq || k != kpad) {
        p = dq_configs[shader_ಗಾತ್ರ].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_ಗಾತ್ರ);
    }

    ಆದರೆ (p == ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::ಸಾಲು_ಅಂತ್ಯ;
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t x_sz = ಗಾತ್ರ(ದಶಮಾಂಶ) * x_ne;
    ಸ್ಥಿರಾಂಕ size_t y_sz = ಗಾತ್ರ(ದಶಮಾಂಶ) * y_ne;
    ಸ್ಥಿರಾಂಕ size_t qx_sz = x_ne * ggml_type_ಗಾತ್ರ(quant)/ggml_blck_ಗಾತ್ರ(quant);
    ಸ್ಥಿರಾಂಕ size_t qy_sz = mmq ? y_ne * ggml_type_ಗಾತ್ರ(GGML_TYPE_Q8_1)/ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_1) : y_sz;
    ಸ್ಥಿರಾಂಕ size_t d_sz = ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne;
    ದಶಮಾಂಶ * x = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(x_sz);
    ದಶಮಾಂಶ * y = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(y_sz);
    ಶೂನ್ಯ * qx = ಸ್ಮರಣೆ_ಕೊಡು(qx_sz);
    vk_ಬಫರ್ qx_buf = ggml_vk_create_ಬಫರ್_check(ctx->device, qx_sz, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    vk_ಬಫರ್ y_buf = ggml_vk_create_ಬಫರ್_check(ctx->device, y_sz, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    vk_ಬಫರ್ qy_buf = ggml_vk_create_ಬಫರ್_check(ctx->device, qy_sz, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    vk_ಬಫರ್ d_buf = ggml_vk_create_ಬಫರ್_check(ctx->device, d_sz, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
    ದಶಮಾಂಶ * d = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(d_sz);
    ದಶಮಾಂಶ * d_chk = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(d_sz);

    ಗಾಗಿ (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (ದಶಮಾಂಶ)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_ದತ್ತಾಂಶ(x, qx, x_ne, quant);

    ಗಾಗಿ (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (ದಶಮಾಂಶ)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    ಆದರೆ (ವಿಭಜಿಸು_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ವಿಭಜಿಸು_k_reduce, ಅಂಕೆ_it);

        ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k == ಶೂನ್ಯಸೂಚಕ || ctx->prealloc_ವಿಭಜಿಸು_k->size < ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k) {
            // Resize buffer
            ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k != ಶೂನ್ಯಸೂಚಕ) {
                ggml_vk_destroy_ಬಫರ್(ctx->prealloc_ವಿಭಜಿಸು_k);
            }
            ctx->prealloc_ವಿಭಜಿಸು_k = ggml_vk_create_ಬಫರ್_check(ctx->device, ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k, {vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal});
        }
    }
    ಆದರೆ (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, ಅಂಕೆ_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_ಬಫರ್_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_ಬಫರ್_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ಆದರೆ (mmq) {
        ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_ವಿಭಜಿಸು_k, 0, ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                ವಿಭಜಿಸು_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_1)) * ggml_type_ಗಾತ್ರ(GGML_TYPE_Q8_1), m * ಗಾತ್ರ(ದಶಮಾಂಶ)
            );
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_ವಿಭಜಿಸು_k, 0, ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                ವಿಭಜಿಸು_k, batch, batch, batch, 1, 1, n,
                k * ಗಾತ್ರ(ದಶಮಾಂಶ), m * ಗಾತ್ರ(ದಶಮಾಂಶ)
            );
        }
    }
    ggml_vk_ctx_ಕೊನೆ(subctx);

    ಸ್ವಯಂಚಾಲಿತ begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ನಿಜ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_ಸರತಿ_command_pools_cleanup(ctx->device);

    ಸ್ವಯಂಚಾಲಿತ end = std::chrono::high_resolution_clock::now();

    ದ್ವಿಗುಣ ಸಮಯ_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_ಬಫರ್_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->ದತ್ತಾಂಶ = qx;
    src1_ggml->ದತ್ತಾಂಶ = y;
    tensor_ggml->ದತ್ತಾಂಶ = d_chk;

    ggml_cgraph * cgraph = ggml_new_ನಕ್ಷೆ(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_ನಕ್ಷೆ_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ದ್ವಿಗುಣ avg_err = 0.0;
    ಪೂರ್ಣಾಂಕ ಮೊದಲ_err_n = -1;
    ಪೂರ್ಣಾಂಕ ಮೊದಲ_err_m = -1;
    ಪೂರ್ಣಾಂಕ ಮೊದಲ_err_b = -1;

    ಗಾಗಿ (size_t i = 0; i < m*n*batch; i++) {
        ದ್ವಿಗುಣ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ಆದರೆ ((err > 0.05f || std::isnan(err)) && ಮೊದಲ_err_n == -1) {
            ಮೊದಲ_err_b = i / (m * n);
            ಮೊದಲ_err_n = (i % (m * n)) / m;
            ಮೊದಲ_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ದ್ವಿಗುಣ tflops = 2.0*m*n*k*batch*ಅಂಕೆ_it / (ಸಮಯ_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    ಆದರೆ (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << ವಿಭಜಿಸು_k << " matmul " << ಸಮಯ_ms / ಅಂಕೆ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::ಸಾಲು_ಅಂತ್ಯ;

    ಆದರೆ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << ಮೊದಲ_err_m << " n = " << ಮೊದಲ_err_n << " b = " << ಮೊದಲ_err_b << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "Actual result: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_ಮಾತೃಕೆ_area(d, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "Expected result: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_ಮಾತೃಕೆ_area(d_chk, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

        std::cerr << "src0: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_ಮಾತೃಕೆ_area(x, GGML_TYPE_F32, k, m, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "src1: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_ಮಾತೃಕೆ_area(y, GGML_TYPE_F32, k, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

        ಆದರೆ (ವಿಭಜಿಸು_k > 1) {
            ದಶಮಾಂಶ * ವಿಭಜಿಸು_k_buf = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k);
            ggml_vk_ಬಫರ್_read(ctx->prealloc_ವಿಭಜಿಸು_k, 0, ವಿಭಜಿಸು_k_buf, ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne * ವಿಭಜಿಸು_k);

            std::cerr << "d_buf0: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            std::cerr << "d_buf1: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf + d_ne, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            std::cerr << "d_buf2: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            std::cerr << "d_buf3: " << std::ಸಾಲು_ಅಂತ್ಯ << std::ಸಾಲು_ಅಂತ್ಯ;
            ggml_vk_print_ಮಾತೃಕೆ_area(ವಿಭಜಿಸು_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, ಮೊದಲ_err_m, ಮೊದಲ_err_n, ಮೊದಲ_err_b);

            ಬಿಡುಗಡೆ(ವಿಭಜಿಸು_k_buf);
        }
    }

    ggml_vk_destroy_ಬಫರ್(qx_buf);
    ggml_vk_destroy_ಬಫರ್(y_buf);
    ggml_vk_destroy_ಬಫರ್(qy_buf);
    ggml_vk_destroy_ಬಫರ್(d_buf);

    ಬಿಡುಗಡೆ(x);
    ಬಿಡುಗಡೆ(qx);
    ಬಿಡುಗಡೆ(y);
    ಬಿಡುಗಡೆ(d);
    ಬಿಡುಗಡೆ(d_chk);
}
#ಆದರೆ_ಅಂತ್ಯ

int64_t ggml_vk_get_op_batch_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಹಿಂತಿರುಗಿ op->ne[1];
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
        ಸ್ಥಿತಿ GGML_OP_ROPE:
        ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ op->ne[2];
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ggml_nrows(op);
    }
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_CHECK_RESULTS
ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_ನಕ್ಷೆ_origin(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *>& ಮುಗಿದ, ಪೂರ್ಣಾಂಕ level = 0) {
    ಆದರೆ (std::find(ಮುಗಿದ.begin(), ಮುಗಿದ.end(), tensor) != ಮುಗಿದ.end() || level > 10) {
        ಹಿಂತಿರುಗಿ;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != ಶೂನ್ಯಸೂಚಕ) << std::ಸಾಲು_ಅಂತ್ಯ;

    ಮುಗಿದ.push_back(tensor);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (tensor->src[i] != ಶೂನ್ಯಸೂಚಕ) {
            ggml_vk_print_ನಕ್ಷೆ_origin(tensor->src[i], ಮುಗಿದ, level + 1);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor_area(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1, ಪೂರ್ಣಾಂಕ i2, ಪೂರ್ಣಾಂಕ i3) {
    ಆದರೆ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ಆದರೆ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                ದಶಮಾಂಶ val;
                ಆದರೆ (tensor->type == GGML_TYPE_F32) {
                    val = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_ಗಮ್ಯ_fp32(*(ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->type == GGML_TYPE_I32) {
                    val = *(ಸ್ಥಿರಾಂಕ int32_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ಇಲ್ಲದಿದ್ದರೆ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಶೂನ್ಯ * tensor_ದತ್ತಾಂಶ = tensor->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_gpu = tensor->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ && ggml_backend_ಬಫರ್_is_vk(tensor->ಬಫರ್);

    ಆದರೆ (is_gpu) {
        ಸ್ಥಿರಾಂಕ size_t tensor_ಗಾತ್ರ = ggml_nbytes(tensor);
        tensor_ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(tensor_ಗಾತ್ರ);

        ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)tensor->ಬಫರ್->context;

        vk_ಬಫರ್ ಬಫರ್_gpu = buf_ctx->dev_ಬಫರ್;
        ggml_vk_ಬಫರ್_read(ಬಫರ್_gpu, vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs, tensor_ದತ್ತಾಂಶ, tensor_ಗಾತ್ರ);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::ಸಾಲು_ಅಂತ್ಯ;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::ಸಾಲು_ಅಂತ್ಯ;
    ಆದರೆ (tensor->src[0] != ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::ಸಾಲು_ಅಂತ್ಯ;
    }
    ಆದರೆ (tensor->src[1] != ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::ಸಾಲು_ಅಂತ್ಯ;
    }
    std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Result:" << std::ಸಾಲು_ಅಂತ್ಯ;
    ggml_vk_print_tensor_area(tensor, tensor_ದತ್ತಾಂಶ, 5, 5, 0, 0);
    std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
    std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> ಮುಗಿದ;
    ggml_vk_print_ನಕ್ಷೆ_origin(tensor, ಮುಗಿದ);

    ಆದರೆ (is_gpu) {
        ಬಿಡುಗಡೆ(tensor_ದತ್ತಾಂಶ);
    }
}

ಶೂನ್ಯ * comp_ಫಲಿತಾಂಶ;
size_t comp_ಗಾತ್ರ;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
ಶೂನ್ಯ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->ಅಂಕೆ_additional_fused_ops];
    ಆದರೆ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    check_counter++;
    ಆದರೆ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        ಹಿಂತಿರುಗಿ;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    ರಚನೆ ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ಸುಳ್ಳು,
    };

    ರಚನೆ ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<ರಚನೆ ggml_tensor *, GGML_MAX_SRC> src_clone = {ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ};
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<ಶೂನ್ಯ *> cloned_mallocs;

    ರಚನೆ ggml_tensor * tensor_clone = ಶೂನ್ಯಸೂಚಕ;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ f = 0; f < ctx->ಅಂಕೆ_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            ಆದರೆ (srci == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            // If a src tensor has been cloned, use that one
            ಸ್ವಯಂಚಾಲಿತ it = cloned_tensors.find(srci);
            ಆದರೆ (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                ಮುಂದುವರಿ;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            size_t srci_ಗಾತ್ರ = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            ಶೂನ್ಯ *src_ಬಫರ್ = ಸ್ಮರಣೆ_ಕೊಡು(srci_ಗಾತ್ರ);
            cloned_mallocs.push_back(src_ಬಫರ್);

            srci_clone->ದತ್ತಾಂಶ = src_ಬಫರ್;
            ಆದರೆ (ggml_backend_ಬಫರ್_is_host(srci->ಬಫರ್)) {
                ಸ್ಮರಣೆ_ನಕಲು(srci_clone->ದತ್ತಾಂಶ, srci->ದತ್ತಾಂಶ, srci_ಗಾತ್ರ);
                ಸ್ಮರಣೆ_ನಕಲು(srci_clone->nb, srci->nb, ಗಾತ್ರ(size_t) * GGML_MAX_DIMS);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_backend_ಬಫರ್_is_vk(srci->ಬಫರ್)) {
                ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)srci->ಬಫರ್->context;
                vk_ಬಫರ್& ಬಫರ್_gpu = buf_ctx->dev_ಬಫರ್;
                uint64_t ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(srci) + srci->view_offs;
                ಆದರೆ (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < srci->ne[3]; i3++) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < srci->ne[2]; i2++) {
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = i3*srci->ne[2] + i2;
                            ggml_vk_ಬಫರ್_read(ಬಫರ್_gpu, ಸ್ಥಾನಾಂತರ + idx * srci->nb[2], ((ಅಕ್ಷರ *)srci_clone->ದತ್ತಾಂಶ + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆದರೆ (ಸ್ಥಾನಾಂತರ + srci_ಗಾತ್ರ >= ಬಫರ್_gpu->size) {
                        srci_ಗಾತ್ರ = ಬಫರ್_gpu->size - ಸ್ಥಾನಾಂತರ;
                    }
                    ggml_vk_ಬಫರ್_read(ಬಫರ್_gpu, ಸ್ಥಾನಾಂತರ, srci_clone->ದತ್ತಾಂಶ, srci_ಗಾತ್ರ);
                    ಸ್ಮರಣೆ_ನಕಲು(srci_clone->nb, srci->nb, ಗಾತ್ರ(size_t) * GGML_MAX_DIMS);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                GGML_ABORT("fatal error");
            }

            ಆದರೆ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        ಆದರೆ (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            ಆದರೆ (src_clone[4]) {
                ggml_flash_attn_ext_ಕೂಡಿಸು_sinks(tensor_clone, src_clone[4]);
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(ಪೂರ್ಣಾಂಕ *)tensor->op_params);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_ಬಹುಲಕ) tensor->op_params[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SCALE) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ARANGE) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ start = ggml_get_op_params_f32(tensor, 0);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ stop = ggml_get_op_params_f32(tensor, 1);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಹೆಜ್ಜೆ = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, ಹೆಜ್ಜೆ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_FILL) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CLAMP) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_REPEAT_ಹಿಂಭಾಗ) {
            tensor_clone = ggml_repeat_ಹಿಂಭಾಗ(ggml_ctx, src_clone[0], tensor);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಕೂಡಿಸು) {
            tensor_clone = ggml_ಕೂಡಿಸು(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಗಣ) {
            tensor_clone = ggml_ಗಣ(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(ದಶಮಾಂಶ *)tensor->op_params);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಗುಂಪು_NORM) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * float_params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)tensor->op_params;
            tensor_clone = ggml_ಗುಂಪು_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(ದಶಮಾಂಶ *)tensor->op_params);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_RMS_NORM_ಹಿಂಭಾಗ) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps = ((ದಶಮಾಂಶ *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_ಹಿಂಭಾಗ(ggml_ctx, src_clone[0], src_clone[1], eps);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SILU_ಹಿಂಭಾಗ) {
            tensor_clone = ggml_silu_ಹಿಂಭಾಗ(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_L2_NORM) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps = ((ದಶಮಾಂಶ *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SOFT_MAX) {
            ಆದರೆ (tensor->src[1] != ಶೂನ್ಯಸೂಚಕ) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } ಇಲ್ಲದಿದ್ದರೆ {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ) {
            tensor_clone = ggml_soft_max_ext_ಹಿಂಭಾಗ(ggml_ctx, src_clone[0], src_clone[1], ((ದಶಮಾಂಶ *)tensor->op_params)[0], ((ದಶಮಾಂಶ *)tensor->op_params)[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_DIAG_ಮುಖವಾಡ_INF) {
            tensor_clone = ggml_diag_ಮುಖವಾಡ_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_ಹಿಂಭಾಗ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims      = ((int32_t *) tensor->op_params)[1];
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ತಳ       = ((ದಶಮಾಂಶ *) tensor->op_params)[5];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale      = ((ದಶಮಾಂಶ *) tensor->op_params)[6];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ      = ((ದಶಮಾಂಶ *) tensor->op_params)[7];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ     = ((ದಶಮಾಂಶ *) tensor->op_params)[8];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_fast       = ((ದಶಮಾಂಶ *) tensor->op_params)[9];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_slow       = ((ದಶಮಾಂಶ *) tensor->op_params)[10];
            ಆದರೆ (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                ಆದರೆ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    tensor_clone = ggml_rope_multi_ಹಿಂಭಾಗ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    tensor_clone = ggml_rope_ext_ಹಿಂಭಾಗ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow);
                }
            }
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs = ((int32_t *) tensor->op_params)[15];
            ಆದರೆ (n_offs != 0) {
                tensor_clone = ggml_rope_ಗಣ_ಸ್ಥಾನಾಂತರ(tensor_clone, n_offs);
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_UNARY) {
            ಆಯ್ಕೆ (ggml_get_unary_op(tensor)) {
            ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                tensor_clone = ggml_gelu_ಶೀಘ್ರ(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_ಗಣ_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_ಗಣ_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_ಗಣ_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_ಗಣ_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                tensor_clone = ggml_ಹೆಜ್ಜೆ(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                tensor_clone = ggml_ಸುತ್ತು(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::ಸಾಲು_ಅಂತ್ಯ;
                GGML_ABORT("fatal error");
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_GLU) {
            ಆದರೆ (src_clone[1] == ಶೂನ್ಯಸೂಚಕ) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } ಇಲ್ಲದಿದ್ದರೆ {
                tensor_clone = ggml_glu_ವಿಭಜಿಸು(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_ಗಣ_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_ಗಣ_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            ಆದರೆ (tensor->src[1] == ಶೂನ್ಯಸೂಚಕ) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } ಇಲ್ಲದಿದ್ದರೆ {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_ಕ್ರಮಬದಲಾವಣೆ(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು) {
            tensor_clone = ggml_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ) *(ಪೂರ್ಣಾಂಕ *)tensor->op_params);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ತುದಿ_K) {
            tensor_clone = ggml_ತುದಿ_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಮೊತ್ತ) {
            tensor_clone = ggml_ಮೊತ್ತ(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು) {
            tensor_clone = ggml_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_DSV4_HC_PRE) {
            ಆದರೆ (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } ಇಲ್ಲದಿದ್ದರೆ {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಸರಾಸರಿ) {
            tensor_clone = ggml_ಸರಾಸರಿ(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ) {
            tensor_clone = ggml_cross_entropy_loss_ಹಿಂಭಾಗ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಎಣಿಕೆ_EQUAL) {
            tensor_clone = ggml_ಎಣಿಕೆ_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_IM2_ಲಂಬಸಾಲು) {
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t s1 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t p1 = tensor->op_params[3];
            ಸ್ಥಿರಾಂಕ int32_t d0 = tensor->op_params[4];
            ಸ್ಥಿರಾಂಕ int32_t d1 = tensor->op_params[5];

            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_IM2_ಲಂಬಸಾಲು_3_D) {
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t s1 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t s2 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[3];
            ಸ್ಥಿರಾಂಕ int32_t p1 = tensor->op_params[4];
            ಸ್ಥಿರಾಂಕ int32_t p2 = tensor->op_params[5];
            ಸ್ಥಿರಾಂಕ int32_t d0 = tensor->op_params[6];
            ಸ್ಥಿರಾಂಕ int32_t d1 = tensor->op_params[7];
            ಸ್ಥಿರಾಂಕ int32_t d2 = tensor->op_params[8];
            ಸ್ಥಿರಾಂಕ int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            ಸ್ಥಿರಾಂಕ int32_t dim = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_COL2IM_1D) {
            ಸ್ಥಿರಾಂಕ int32_t stride = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t oc     = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_POOL_1D) {
            ಎಣಿಕೆ ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ಸ್ಥಿರಾಂಕ int32_t k0 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_POOL_2D) {
            ಎಣಿಕೆ ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ಸ್ಥಿರಾಂಕ int32_t k0 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t k1 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[3];
            ಸ್ಥಿರಾಂಕ int32_t s1 = tensor->op_params[4];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[5];
            ಸ್ಥಿರಾಂಕ int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CONV_2D) {
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t s1 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t p1 = tensor->op_params[3];
            ಸ್ಥಿರಾಂಕ int32_t d0 = tensor->op_params[4];
            ಸ್ಥಿರಾಂಕ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CONV_3D) {
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t s1 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t s2 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[3];
            ಸ್ಥಿರಾಂಕ int32_t p1 = tensor->op_params[4];
            ಸ್ಥಿರಾಂಕ int32_t p2 = tensor->op_params[5];
            ಸ್ಥಿರಾಂಕ int32_t d0 = tensor->op_params[6];
            ಸ್ಥಿರಾಂಕ int32_t d1 = tensor->op_params[7];
            ಸ್ಥಿರಾಂಕ int32_t d2 = tensor->op_params[8];
            ಸ್ಥಿರಾಂಕ int32_t IC = tensor->op_params[9];
            ಸ್ಥಿರಾಂಕ int32_t N  = tensor->op_params[10];
            ಸ್ಥಿರಾಂಕ int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_ನೇರ(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CONV_2D_DW) {
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t s1 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t p0 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t p1 = tensor->op_params[3];
            ಸ್ಥಿರಾಂಕ int32_t d0 = tensor->op_params[4];
            ಸ್ಥಿರಾಂಕ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_ನೇರ(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            ಸ್ಥಿರಾಂಕ int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_LEAKY_RELU) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * op_params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], ಸುಳ್ಳು);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_GATED_ರೇಖೀಯ_ATTN) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * op_params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)tensor->op_params;
            tensor_clone = ggml_gated_ರೇಖೀಯ_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_ಹೆಜ್ಜೆ_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_ಹೆಜ್ಜೆ_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ಕೂಡಿಸು_ID) {
            tensor_clone = ggml_ಕೂಡಿಸು_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SSM_ಪರಿಶೀಲನೆ) {
            ಸ್ಥಿರಾಂಕ int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_ಪರಿಶೀಲನೆ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->op == GGML_OP_ROLL) {
            ಸ್ಥಿರಾಂಕ int32_t s0 = tensor->op_params[0];
            ಸ್ಥಿರಾಂಕ int32_t s1 = tensor->op_params[1];
            ಸ್ಥಿರಾಂಕ int32_t s2 = tensor->op_params[2];
            ಸ್ಥಿರಾಂಕ int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        ಇಲ್ಲದಿದ್ದರೆ {
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::ಸಾಲು_ಅಂತ್ಯ;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_ನಕ್ಷೆ(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_ನಕ್ಷೆ_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    ಆದರೆ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_ಗಾತ್ರ = ggml_nbytes(tensor_clone);

    comp_ಫಲಿತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(comp_ಗಾತ್ರ);
    ಸ್ಮರಣೆ_ನಕಲು(comp_ಫಲಿತಾಂಶ, tensor_clone->ದತ್ತಾಂಶ, comp_ಗಾತ್ರ);
    ಸ್ಮರಣೆ_ನಕಲು(comp_nb, tensor_clone->nb, ಗಾತ್ರ(size_t) * GGML_MAX_DIMS);

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ m : cloned_mallocs) {
        ಬಿಡುಗಡೆ(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

ಶೂನ್ಯ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->ಅಂಕೆ_additional_fused_ops];
    ಆದರೆ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        ಹಿಂತಿರುಗಿ;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    ಶೂನ್ಯ * tensor_ದತ್ತಾಂಶ = tensor->ದತ್ತಾಂಶ;

    ಆದರೆ (ggml_backend_ಬಫರ್_is_vk(tensor->ಬಫರ್)) {
        size_t tensor_ಗಾತ್ರ = ggml_nbytes(tensor);
        tensor_ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(tensor_ಗಾತ್ರ);

        ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)tensor->ಬಫರ್->context;

        vk_ಬಫರ್& ಬಫರ್_gpu = buf_ctx->dev_ಬಫರ್;
        uint64_t ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs;
        ಆದರೆ (ಸ್ಥಾನಾಂತರ + tensor_ಗಾತ್ರ >= ಬಫರ್_gpu->size) {
            tensor_ಗಾತ್ರ = ಬಫರ್_gpu->size - ಸ್ಥಾನಾಂತರ;
        }

        ggml_vk_ಬಫರ್_read(ಬಫರ್_gpu, ಸ್ಥಾನಾಂತರ, tensor_ದತ್ತಾಂಶ, tensor_ಗಾತ್ರ);
    }

    ದಶಮಾಂಶ ಮೊದಲ_error_ಫಲಿತಾಂಶ = -1.0f;
    ದಶಮಾಂಶ ಮೊದಲ_error_correct = -1.0f;
    std::array<ಪೂರ್ಣಾಂಕ, 4> ಮೊದಲ_error = { -1, -1, -1, -1 };
    ದ್ವಿಗುಣ avg_err = 0.0;
    size_t counter = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i3 = 0; i3 < tensor->ne[3]; i3++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i2 = 0; i2 < tensor->ne[2]; i2++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i1 = 0; i1 < tensor->ne[1]; i1++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i0 = 0; i0 < tensor->ne[0]; i0++) {
                    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಬಫರ್_ಗಾತ್ರ_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_ಗಾತ್ರ;
                    ದಶಮಾಂಶ correct = 0.0f;
                    ದಶಮಾಂಶ ಫಲಿತಾಂಶ = 0.0f;

                    ಆದರೆ (ಬಫರ್_ಗಾತ್ರ_fit) {
                        ಆದರೆ (tensor->type == GGML_TYPE_F32) {
                            correct = *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) comp_ಫಲಿತಾಂಶ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ಫಲಿತಾಂಶ  = *(ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) tensor_ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_ಗಮ್ಯ_fp32(*(ggml_fp16_t *) ((ಅಕ್ಷರ *) comp_ಫಲಿತಾಂಶ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ಫಲಿತಾಂಶ  = ggml_fp16_ಗಮ್ಯ_fp32(*(ggml_fp16_t *) ((ಅಕ್ಷರ *) tensor_ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_ಗಮ್ಯ_fp32(*(ggml_bf16_t *) ((ಅಕ್ಷರ *) comp_ಫಲಿತಾಂಶ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ಫಲಿತಾಂಶ  = ggml_bf16_ಗಮ್ಯ_fp32(*(ggml_bf16_t *) ((ಅಕ್ಷರ *) tensor_ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((ಅಕ್ಷರ *) comp_ಫಲಿತಾಂಶ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ಫಲಿತಾಂಶ  = *(int32_t *) ((ಅಕ್ಷರ *) tensor_ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((ಅಕ್ಷರ *) comp_ಫಲಿತಾಂಶ + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ಫಲಿತಾಂಶ  = *(int64_t *) ((ಅಕ್ಷರ *) tensor_ದತ್ತಾಂಶ + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::ಸಾಲು_ಅಂತ್ಯ;
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::ಸಾಲು_ಅಂತ್ಯ;
                        GGML_ABORT("fatal error");
                    }

                    ಆದರೆ ((std::isnan(correct) != std::isnan(ಫಲಿತಾಂಶ)) || (std::isinf(correct) != std::isinf(ಫಲಿತಾಂಶ)) || !ಬಫರ್_ಗಾತ್ರ_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << ಫಲಿತಾಂಶ << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::ಸಾಲು_ಅಂತ್ಯ;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
                        ಆದರೆ (src0 != ಶೂನ್ಯಸೂಚಕ) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
                        }
                        ಆದರೆ (src1 != ಶೂನ್ಯಸೂಚಕ) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
                        }
                        ಆದರೆ (src2 != ಶೂನ್ಯಸೂಚಕ) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
                        }
                        ಆದರೆ (src3 != ಶೂನ್ಯಸೂಚಕ) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
                        }
                        std::cerr << "First error: result=" << ಮೊದಲ_error_ಫಲಿತಾಂಶ << " correct=" << ಮೊದಲ_error_correct  << " i3=" << ಮೊದಲ_error[3] << " i2=" << ಮೊದಲ_error[2] << " i1=" << ಮೊದಲ_error[1] << " i0=" << ಮೊದಲ_error[0] << std::ಸಾಲು_ಅಂತ್ಯ;
                        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Result:" << std::ಸಾಲು_ಅಂತ್ಯ;
                        ggml_vk_print_tensor_area(tensor, tensor_ದತ್ತಾಂಶ, i0, i1, i2, i3);
                        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Correct:" << std::ಸಾಲು_ಅಂತ್ಯ;
                        ggml_vk_print_tensor_area(tensor, comp_ಫಲಿತಾಂಶ, i0, i1, i2, i3);
                        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
                        std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> ಮುಗಿದ;
                        ggml_vk_print_ನಕ್ಷೆ_origin(tensor, ಮುಗಿದ);
                        GGML_ABORT("fatal error");
                    }
                    ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    ಆದರೆ (ಮೊದಲ_error[0] == -1 && std::fabs(correct - ಫಲಿತಾಂಶ) / denom > 0.5) {
                        ಮೊದಲ_error[0] = i0;
                        ಮೊದಲ_error[1] = i1;
                        ಮೊದಲ_error[2] = i2;
                        ಮೊದಲ_error[3] = i3;
                        ಮೊದಲ_error_ಫಲಿತಾಂಶ = ಫಲಿತಾಂಶ;
                        ಮೊದಲ_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    ಆದರೆ (!std::isinf(correct) && !std::isinf(ಫಲಿತಾಂಶ) && !std::isnan(correct) && !std::isnan(ಫಲಿತಾಂಶ)) {
                        avg_err += std::fabs(correct - ಫಲಿತಾಂಶ) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    ಆದರೆ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        ಆದರೆ (src0 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        ಆದರೆ (src1 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        ಆದರೆ (src2 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        ಆದರೆ (src3 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        std::cerr << "First error: result=" << ಮೊದಲ_error_ಫಲಿತಾಂಶ << " correct=" << ಮೊದಲ_error_correct  << " i3=" << ಮೊದಲ_error[3] << " i2=" << ಮೊದಲ_error[2] << " i1=" << ಮೊದಲ_error[1] << " i0=" << ಮೊದಲ_error[0] << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Result:" << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_tensor_area(tensor, tensor_ದತ್ತಾಂಶ, 5, 5, 0, 0);
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Correct:" << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_tensor_area(tensor, comp_ಫಲಿತಾಂಶ, 5, 5, 0, 0);
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
        std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> ಮುಗಿದ;
        ggml_vk_print_ನಕ್ಷೆ_origin(tensor, ಮುಗಿದ);
    }

    ಆದರೆ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        ಆದರೆ (src0 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        ಆದರೆ (src1 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        ಆದರೆ (src2 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        ಆದರೆ (src3 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::ಸಾಲು_ಅಂತ್ಯ;
        }
        std::cerr << "First error: result=" << ಮೊದಲ_error_ಫಲಿತಾಂಶ << " correct=" << ಮೊದಲ_error_correct  << " i3=" << ಮೊದಲ_error[3] << " i2=" << ಮೊದಲ_error[2] << " i1=" << ಮೊದಲ_error[1] << " i0=" << ಮೊದಲ_error[0] << std::ಸಾಲು_ಅಂತ್ಯ;
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Result:" << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_tensor_area(tensor, tensor_ದತ್ತಾಂಶ, ಮೊದಲ_error[0], ಮೊದಲ_error[1], ಮೊದಲ_error[2], ಮೊದಲ_error[3]);
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ << "Correct:" << std::ಸಾಲು_ಅಂತ್ಯ;
        ggml_vk_print_tensor_area(tensor, comp_ಫಲಿತಾಂಶ, ಮೊದಲ_error[0], ಮೊದಲ_error[1], ಮೊದಲ_error[2], ಮೊದಲ_error[3]);
        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
        std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> ಮುಗಿದ;
        ggml_vk_print_ನಕ್ಷೆ_origin(tensor, ಮುಗಿದ);
        GGML_ABORT("fatal error");
    } ಇಲ್ಲದಿದ್ದರೆ {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::ಸಾಲು_ಅಂತ್ಯ;
    }

    ಬಿಡುಗಡೆ(comp_ಫಲಿತಾಂಶ);
    comp_ಫಲಿತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
    comp_ಗಾತ್ರ = 0;

    ಆದರೆ (ggml_backend_ಬಫರ್_is_vk(tensor->ಬಫರ್)) {
        ಬಿಡುಗಡೆ(tensor_ದತ್ತಾಂಶ);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#ಆದರೆ_ಅಂತ್ಯ

