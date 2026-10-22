#ಅನ್ತರ್ಭಾವ "ggml-vulkan-common.h"

ತಾರ್ಕಿಕ vk_memory_logger_enabled = ಅಸತ್ಯ;

ತಾರ್ಕಿಕ vk_perf_logger_enabled = ಅಸತ್ಯ;

ತಾರ್ಕಿಕ vk_perf_logger_concurrent = ಅಸತ್ಯ;

ತಾರ್ಕಿಕ vk_enable_sync_logger = ಅಸತ್ಯ;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

ಶೂನ್ಯ vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
    ಯದಿ (!vk_memory_logger_enabled) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    ಧ್ರುವ ತಾರ್ಕಿಕ device = ತಾರ್ಕಿಕ(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    ಧ್ರುವ std::string type = device ? "device" : "host";
    allocations[buf->buffer] = size;
    total_device += device ? size : 0;
    total_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
}

ಶೂನ್ಯ vk_memory_logger::log_deallocation(vk_buffer_ref buf_ref) {
    ಯದಿ (buf_ref.expired() || buf_ref.lock()->size == 0 || !vk_memory_logger_enabled) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    ಧ್ರುವ ತಾರ್ಕಿಕ device = ತಾರ್ಕಿಕ(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::string type = device ? "device" : "host";
    auto it = allocations.find(buf->buffer);
    ಯದಿ (it != allocations.end()) {
        total_device -= device ? it->second : 0;
        total_host -= device ? 0 : it->second;
        VK_LOG_MEMORY(buf->device->name << ": -" << format_size(it->second) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
        allocations.erase(it);
    } ಅನ್ಯಥಾ {
        VK_LOG_MEMORY("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->buffer);
    }
}

#ifdef GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ವರ್ಣ * name);
#endif

#ifdef GGML_VULKAN_RUN_TESTS
ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_matrix_area(ಧ್ರುವ ಶೂನ್ಯ * data, ggml_type type, ಪೂರ್ಣಾಙ್ಕ ne0, ಪೂರ್ಣಾಙ್ಕ ne1, ಪೂರ್ಣಾಙ್ಕ i0, ಪೂರ್ಣಾಙ್ಕ i1, ಪೂರ್ಣಾಙ್ಕ i2) {
    ಯದಿ (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ಯದಿ (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                ಭಿನ್ನ val;
                ಯದಿ (type == GGML_TYPE_F32) {
                    val = *((ಧ್ರುವ ಭಿನ್ನ *) data + i2*ne1*ne0 + idx1*ne0 + idx0);
                } ಅನ್ಯಥಾ ಯದಿ (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*((ಧ್ರುವ ggml_fp16_t *) data + i2*ne1*ne0 + idx1*ne0 + idx0));
                } ಅನ್ಯಥಾ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ಅನ್ಯಥಾ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

template <typename X_TYPE, typename Y_TYPE>
ಶೂನ್ಯ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, ಪೂರ್ಣಾಙ್ಕ split_k, ಪೂರ್ಣಾಙ್ಕ shader_size) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << ಅಂಕೆ_it << ", " << split_k << ", " << shader_size << ")");
    ಧ್ರುವ size_t x_ne = m * k * batch;
    ಧ್ರುವ size_t y_ne = k * n * batch;
    ಧ್ರುವ size_t d_ne = m * n * batch;

    ggml_type x_type = std::is_same<ಭಿನ್ನ, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<ಭಿನ್ನ, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_ಕೀಲಿ mm_test_ಕೀಲಿ{x_type, y_type, ಅಸತ್ಯ, ಅಸತ್ಯ};
    auto mm_test_it = ctx->device->pipeline_matmul.find(mm_test_ಕೀಲಿ);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.empty());
    auto& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_size >= 0 && shader_size < (ಪೂರ್ಣಾಙ್ಕ)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_size);
    vk_pipeline p = mm_test_configs[shader_size].aligned ? mm_test_configs[shader_size].aligned : mm_test_configs[shader_size].unaligned;

    ಧ್ರುವ size_t kpad = ggml_vk_align_size(k, mm_test_configs[shader_size].align);

    ಯದಿ (k != kpad) {
        p = mm_test_configs[shader_size].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_size);
    }

    ಯದಿ (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, ಅಂಕೆ_it);

        ಯದಿ (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k) {
            // Resize buffer
            ಯದಿ (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_buffer d_X = ggml_vk_create_buffer_check(ctx->device, ಪರಿಮಾಣ(X_TYPE) * x_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_Y = ggml_vk_create_buffer_check(ctx->device, ಪರಿಮಾಣ(Y_TYPE) * y_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_D = ggml_vk_create_buffer_check(ctx->device, ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) malloc(ಪರಿಮಾಣ(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) malloc(ಪರಿಮಾಣ(Y_TYPE) * y_ne);
    ಭಿನ್ನ* d = (ಭಿನ್ನ *) malloc(ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne);

    ಪ್ರತಿ (size_t i = 0; i < x_ne; i++) {
        ಯದಿ (std::is_same<ಭಿನ್ನ, X_TYPE>()) {
            x[i] = (rand() / (ಭಿನ್ನ)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } ಅನ್ಯಥಾ ಯದಿ (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_to_fp16((rand() / (ಭಿನ್ನ)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("fatal error");
        }
    }
    ಪ್ರತಿ (size_t i = 0; i < y_ne; i++) {
        ಯದಿ (std::is_same<ಭಿನ್ನ, Y_TYPE>()) {
            y[i] = (rand() / (ಭಿನ್ನ)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } ಅನ್ಯಥಾ ಯದಿ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_to_fp16((rand() / (ಭಿನ್ನ)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_buffer_write(d_X, 0, x, ಪರಿಮಾಣ(X_TYPE) * k * m * batch);
    ggml_vk_buffer_write(d_Y, 0, y, ಪರಿಮಾಣ(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ಪ್ರತಿ (size_t i = 0; i < ಅಂಕೆ_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1, n,
            k * ಪರಿಮಾಣ(Y_TYPE), m * ಪರಿಮಾಣ(ಭಿನ್ನ)
        );
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ಸತ್ಯ, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();
    ದ್ವಿಗುಣ time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_buffer_read(d_D, 0, d, ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne);

    ಭಿನ್ನ * d_chk = (ಭಿನ್ನ *) malloc(ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    ಯದಿ (std::is_same<ಭಿನ್ನ, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } ಅನ್ಯಥಾ ಯದಿ (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } ಅನ್ಯಥಾ {
        GGML_ABORT("fatal error");
    }
    ಯದಿ (std::is_same<ಭಿನ್ನ, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } ಅನ್ಯಥಾ ಯದಿ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } ಅನ್ಯಥಾ {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = x;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ದ್ವಿಗುಣ avg_err = 0.0;
    ಪೂರ್ಣಾಙ್ಕ first_err_n = -1;
    ಪೂರ್ಣಾಙ್ಕ first_err_m = -1;
    ಪೂರ್ಣಾಙ್ಕ first_err_b = -1;

    ಪ್ರತಿ (size_t i = 0; i < m*n*batch; i++) {
        ದ್ವಿಗುಣ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ಯದಿ ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ದ್ವಿಗುಣ tflops = 2.0*m*n*k*batch*ಅಂಕೆ_it / (time / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time / ಅಂಕೆ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    ಯದಿ (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        ಯದಿ (split_k > 1) {
            ಭಿನ್ನ * split_k_buf = (ಭಿನ್ನ *) malloc(ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            ಮುಕ್ತ(split_k_buf);
        }
    }

    ಮುಕ್ತ(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_buffer(d_X);
    ggml_vk_destroy_buffer(d_Y);
    ggml_vk_destroy_buffer(d_D);

    ಮುಕ್ತ(x);
    ಮುಕ್ತ(y);
    ಮುಕ್ತ(d);
}

template ಶೂನ್ಯ ggml_vk_test_matmul<ggml_fp16_t, ಭಿನ್ನ>(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, ಪೂರ್ಣಾಙ್ಕ split_k, ಪೂರ್ಣಾಙ್ಕ shader_size);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor_area(ಧ್ರುವ ggml_tensor * tensor, ಪೂರ್ಣಾಙ್ಕ i0, ಪೂರ್ಣಾಙ್ಕ i1, ಪೂರ್ಣಾಙ್ಕ i2, ಪೂರ್ಣಾಙ್ಕ i3) {
    ಯದಿ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ಯದಿ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                ಭಿನ್ನ val;
                ಯದಿ (tensor->type == GGML_TYPE_F32) {
                    val = *(ಭಿನ್ನ *) ((ವರ್ಣ *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ಅನ್ಯಥಾ ಯದಿ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((ವರ್ಣ *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } ಅನ್ಯಥಾ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ಅನ್ಯಥಾ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_quantize_data(ಧ್ರುವ ಭಿನ್ನ * from, ಶೂನ್ಯ * to, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, to, 0, 1, ne, nullptr);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_dequantize_data(ಧ್ರುವ ಶೂನ್ಯ * from, ಭಿನ್ನ * to, size_t ne, ggml_type quant) {
    ಯದಿ (quant == GGML_TYPE_F32) {
        memcpy(to, from, ಪರಿಮಾಣ(ಭಿನ್ನ) * ne);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ auto * tt = ggml_get_type_traits(quant);

    ggml_to_float_t dequant_fn = tt->to_float;

    dequant_fn(from, to, ne);
}

ಶೂನ್ಯ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    ಧ್ರುವ size_t x_sz = ಪರಿಮಾಣ(ಭಿನ್ನ) * ne;
    ಧ್ರುವ size_t x_sz_f16 = ಪರಿಮಾಣ(ggml_fp16_t) * ne;
    ಧ್ರುವ size_t qx_sz = ne * ggml_type_size(quant)/ggml_blck_size(quant);
    ಭಿನ್ನ * x = (ಭಿನ್ನ *) malloc(x_sz);
    ಶೂನ್ಯ * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer x_buf = ggml_vk_create_buffer_check(ctx->device, x_sz_f16, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    ಭಿನ್ನ * x_ref = (ಭಿನ್ನ *) malloc(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) malloc(x_sz_f16);

    ಪ್ರತಿ (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (ಭಿನ್ನ)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_to_fp16(ctx, quant);

    ggml_vk_quantize_data(x, qx, ne, quant);
    ggml_vk_dequantize_data(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ಧ್ರುವ std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ಸತ್ಯ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    ದ್ವಿಗುಣ ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(x_buf, 0, x_chk, x_sz_f16);

    ಪೂರ್ಣಾಙ್ಕ first_err = -1;

    ದ್ವಿಗುಣ avg_err = 0.0;
    ಪ್ರತಿ (size_t i = 0; i < ne; i++) {
        ದ್ವಿಗುಣ error = std::fabs(x_ref[i] - ggml_fp16_to_fp32(x_chk[i]));
        avg_err += error;

        ಯದಿ (first_err < 0 && error > 0.05) {
            first_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::endl;

    ಯದಿ (avg_err > 0.1) {
        std::cerr << "first_error = " << first_err << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = std::max(0, first_err - 5); i < std::min((ಪೂರ್ಣಾಙ್ಕ)ne, first_err + 5); i++) {
            std::cerr << ggml_fp16_to_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = std::max(0, first_err - 5); i < std::min((ಪೂರ್ಣಾಙ್ಕ)ne, first_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::endl;
    }

    ggml_vk_destroy_buffer(x_buf);
    ggml_vk_destroy_buffer(qx_buf);

    ಮುಕ್ತ(x);
    ಮುಕ್ತ(qx);
    ಮುಕ್ತ(x_ref);
    ಮುಕ್ತ(x_chk);
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

ಶೂನ್ಯ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t ಅಂಕೆ_it, size_t split_k, size_t shader_size, ggml_type quant, ತಾರ್ಕಿಕ mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << ಅಂಕೆ_it << ", " << split_k << ", " << ggml_type_name(quant) << ")");
    ಧ್ರುವ size_t x_ne = m * k * batch;
    ಧ್ರುವ size_t y_ne = k * n * batch;
    ಧ್ರುವ size_t d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    ತಾರ್ಕಿಕ f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_ಕೀಲಿ dq_ಕೀಲಿ{quant, b_type, ಅಸತ್ಯ, f16acc};
    auto dq_it = ctx->device->pipeline_matmul.find(dq_ಕೀಲಿ);
    ಯದಿ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        ಯದಿ (f16acc) {
            dq_ಕೀಲಿ.f16acc = ಅಸತ್ಯ;
            dq_it = ctx->device->pipeline_matmul.find(dq_ಕೀಲಿ);
        }
    }
    ಯದಿ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::endl;
        ಪ್ರತ್ಯಾಗಮ;
    }
    auto& dq_configs = dq_it->second;
    ಯದಿ (shader_size >= (ಪೂರ್ಣಾಙ್ಕ)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_size << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::endl;
        ಪ್ರತ್ಯಾಗಮ;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_size);
    vk_pipeline p = dq_configs[shader_size].aligned ? dq_configs[shader_size].aligned : dq_configs[shader_size].unaligned;

    ಧ್ರುವ size_t kpad = mmq ? 0 : ggml_vk_align_size(k, dq_configs[shader_size].align);

    ಯದಿ (mmq || k != kpad) {
        p = dq_configs[shader_size].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_size);
    }

    ಯದಿ (p == nullptr) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::endl;
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ size_t x_sz = ಪರಿಮಾಣ(ಭಿನ್ನ) * x_ne;
    ಧ್ರುವ size_t y_sz = ಪರಿಮಾಣ(ಭಿನ್ನ) * y_ne;
    ಧ್ರುವ size_t qx_sz = x_ne * ggml_type_size(quant)/ggml_blck_size(quant);
    ಧ್ರುವ size_t qy_sz = mmq ? y_ne * ggml_type_size(GGML_TYPE_Q8_1)/ggml_blck_size(GGML_TYPE_Q8_1) : y_sz;
    ಧ್ರುವ size_t d_sz = ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne;
    ಭಿನ್ನ * x = (ಭಿನ್ನ *) malloc(x_sz);
    ಭಿನ್ನ * y = (ಭಿನ್ನ *) malloc(y_sz);
    ಶೂನ್ಯ * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer y_buf = ggml_vk_create_buffer_check(ctx->device, y_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer qy_buf = ggml_vk_create_buffer_check(ctx->device, qy_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_buf = ggml_vk_create_buffer_check(ctx->device, d_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    ಭಿನ್ನ * d = (ಭಿನ್ನ *) malloc(d_sz);
    ಭಿನ್ನ * d_chk = (ಭಿನ್ನ *) malloc(d_sz);

    ಪ್ರತಿ (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (ಭಿನ್ನ)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_data(x, qx, x_ne, quant);

    ಪ್ರತಿ (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (ಭಿನ್ನ)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    ಯದಿ (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, ಅಂಕೆ_it);

        ಯದಿ (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k) {
            // Resize buffer
            ಯದಿ (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }
    ಯದಿ (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, ಅಂಕೆ_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_buffer_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ಯದಿ (mmq) {
        ಪ್ರತಿ (size_t i = 0; i < ಅಂಕೆ_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_size(GGML_TYPE_Q8_1)) * ggml_type_size(GGML_TYPE_Q8_1), m * ಪರಿಮಾಣ(ಭಿನ್ನ)
            );
        }
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (size_t i = 0; i < ಅಂಕೆ_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                k * ಪರಿಮಾಣ(ಭಿನ್ನ), m * ಪರಿಮಾಣ(ಭಿನ್ನ)
            );
        }
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ಸತ್ಯ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    ದ್ವಿಗುಣ time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = qx;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ದ್ವಿಗುಣ avg_err = 0.0;
    ಪೂರ್ಣಾಙ್ಕ first_err_n = -1;
    ಪೂರ್ಣಾಙ್ಕ first_err_m = -1;
    ಪೂರ್ಣಾಙ್ಕ first_err_b = -1;

    ಪ್ರತಿ (size_t i = 0; i < m*n*batch; i++) {
        ದ್ವಿಗುಣ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ಯದಿ ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ದ್ವಿಗುಣ tflops = 2.0*m*n*k*batch*ಅಂಕೆ_it / (time_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    ಯದಿ (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time_ms / ಅಂಕೆ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    ಯದಿ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        std::cerr << "src0: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(x, GGML_TYPE_F32, k, m, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        std::cerr << "src1: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(y, GGML_TYPE_F32, k, n, first_err_m, first_err_n, first_err_b);

        ಯದಿ (split_k > 1) {
            ಭಿನ್ನ * split_k_buf = (ಭಿನ್ನ *) malloc(ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, ಪರಿಮಾಣ(ಭಿನ್ನ) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            ಮುಕ್ತ(split_k_buf);
        }
    }

    ggml_vk_destroy_buffer(qx_buf);
    ggml_vk_destroy_buffer(y_buf);
    ggml_vk_destroy_buffer(qy_buf);
    ggml_vk_destroy_buffer(d_buf);

    ಮುಕ್ತ(x);
    ಮುಕ್ತ(qx);
    ಮುಕ್ತ(y);
    ಮುಕ್ತ(d);
    ಮುಕ್ತ(d_chk);
}
#endif

int64_t ggml_vk_get_op_batch_size(ಧ್ರುವ ggml_tensor * op) {
    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_GET_ROWS:
            ಪ್ರತ್ಯಾಗಮ 0;
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            ಪ್ರತ್ಯಾಗಮ op->ne[1];
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
        ಪಕ್ಷ GGML_OP_ROPE:
        ಪಕ್ಷ GGML_OP_ROPE_BACK:
            ಪ್ರತ್ಯಾಗಮ op->ne[2];
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ggml_nrows(op);
    }
}

#ifdef GGML_VULKAN_CHECK_RESULTS
ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_graph_origin(ಧ್ರುವ ggml_tensor * tensor, std::vector<ಧ್ರುವ ggml_tensor *>& done, ಪೂರ್ಣಾಙ್ಕ level = 0) {
    ಯದಿ (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;

    done.push_back(tensor);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (tensor->src[i] != nullptr) {
            ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor_area(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, ಪೂರ್ಣಾಙ್ಕ i0, ಪೂರ್ಣಾಙ್ಕ i1, ಪೂರ್ಣಾಙ್ಕ i2, ಪೂರ್ಣಾಙ್ಕ i3) {
    ಯದಿ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ಯದಿ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                ಭಿನ್ನ val;
                ಯದಿ (tensor->type == GGML_TYPE_F32) {
                    val = *(ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ಅನ್ಯಥಾ ಯದಿ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ಧ್ರುವ ggml_fp16_t *) ((ಧ್ರುವ ವರ್ಣ *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } ಅನ್ಯಥಾ ಯದಿ (tensor->type == GGML_TYPE_I32) {
                    val = *(ಧ್ರುವ int32_t *) ((ಧ್ರುವ ವರ್ಣ *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } ಅನ್ಯಥಾ {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } ಅನ್ಯಥಾ {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_tensor(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ವರ್ಣ * name) {
    ಶೂನ್ಯ * tensor_data = tensor->data;

    ಧ್ರುವ ತಾರ್ಕಿಕ is_gpu = tensor->buffer != nullptr && ggml_backend_buffer_is_vk(tensor->buffer);

    ಯದಿ (is_gpu) {
        ಧ್ರುವ size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer buffer_gpu = buf_ctx->dev_buffer;
        ggml_vk_buffer_read(buffer_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_data, tensor_size);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::endl;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
    ಯದಿ (tensor->src[0] != nullptr) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::endl;
    }
    ಯದಿ (tensor->src[1] != nullptr) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::endl;
    }
    std::cerr << std::endl << "Result:" << std::endl;
    ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<ಧ್ರುವ ggml_tensor *> done;
    ggml_vk_print_graph_origin(tensor, done);

    ಯದಿ (is_gpu) {
        ಮುಕ್ತ(tensor_data);
    }
}

ಶೂನ್ಯ * comp_result;
size_t comp_size;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
ಶೂನ್ಯ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->ಅಂಕೆ_additional_fused_ops];
    ಯದಿ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    check_counter++;
    ಯದಿ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    ಸಂರಚನಾ ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಅಸತ್ಯ,
    };

    ಸಂರಚನಾ ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<ಸಂರಚನಾ ggml_tensor *, GGML_MAX_SRC> src_clone = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    ಧ್ರುವ ವರ್ಣ * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<ಶೂನ್ಯ *> cloned_mallocs;

    ಸಂರಚನಾ ggml_tensor * tensor_clone = nullptr;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ f = 0; f < ctx->ಅಂಕೆ_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            ಯದಿ (srci == nullptr) {
                ಅನುವರ್ತ;
            }
            // If a src tensor has been cloned, use that one
            auto it = cloned_tensors.find(srci);
            ಯದಿ (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                ಅನುವರ್ತ;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            size_t srci_size = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            ಶೂನ್ಯ *src_buffer = malloc(srci_size);
            cloned_mallocs.push_back(src_buffer);

            srci_clone->data = src_buffer;
            ಯದಿ (ggml_backend_buffer_is_host(srci->buffer)) {
                memcpy(srci_clone->data, srci->data, srci_size);
                memcpy(srci_clone->nb, srci->nb, ಪರಿಮಾಣ(size_t) * GGML_MAX_DIMS);
            } ಅನ್ಯಥಾ ಯದಿ (ggml_backend_buffer_is_vk(srci->buffer)) {
                ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)srci->buffer->context;
                vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
                uint64_t offset = vk_tensor_offset(srci) + srci->view_offs;
                ಯದಿ (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i3 = 0; i3 < srci->ne[3]; i3++) {
                        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i2 = 0; i2 < srci->ne[2]; i2++) {
                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = i3*srci->ne[2] + i2;
                            ggml_vk_buffer_read(buffer_gpu, offset + idx * srci->nb[2], ((ವರ್ಣ *)srci_clone->data + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } ಅನ್ಯಥಾ {
                    ಯದಿ (offset + srci_size >= buffer_gpu->size) {
                        srci_size = buffer_gpu->size - offset;
                    }
                    ggml_vk_buffer_read(buffer_gpu, offset, srci_clone->data, srci_size);
                    memcpy(srci_clone->nb, srci->nb, ಪರಿಮಾಣ(size_t) * GGML_MAX_DIMS);
                }
            } ಅನ್ಯಥಾ {
                GGML_ABORT("fatal error");
            }

            ಯದಿ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        ಯದಿ (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ಧ್ರುವ ಭಿನ್ನ * params = (ಧ್ರುವ ಭಿನ್ನ *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            ಯದಿ (src_clone[4]) {
                ggml_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]);
            }
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(ಪೂರ್ಣಾಙ್ಕ *)tensor->op_params);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_mode) tensor->op_params[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SCALE) {
            ಧ್ರುವ ಭಿನ್ನ * params = (ಧ್ರುವ ಭಿನ್ನ *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ARANGE) {
            ಧ್ರುವ ಭಿನ್ನ start = ggml_get_op_params_f32(tensor, 0);
            ಧ್ರುವ ಭಿನ್ನ stop = ggml_get_op_params_f32(tensor, 1);
            ಧ್ರುವ ಭಿನ್ನ step = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, step);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_FILL) {
            ಧ್ರುವ ಭಿನ್ನ value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CLAMP) {
            ಧ್ರುವ ಭಿನ್ನ * params = (ಧ್ರುವ ಭಿನ್ನ *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_REPEAT_BACK) {
            tensor_clone = ggml_repeat_back(ggml_ctx, src_clone[0], tensor);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ADD) {
            tensor_clone = ggml_add(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SET) {
            tensor_clone = ggml_set(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(ಭಿನ್ನ *)tensor->op_params);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_GROUP_NORM) {
            ಧ್ರುವ ಭಿನ್ನ * float_params = (ಧ್ರುವ ಭಿನ್ನ *)tensor->op_params;
            tensor_clone = ggml_group_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(ಭಿನ್ನ *)tensor->op_params);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_RMS_NORM_BACK) {
            ಧ್ರುವ ಭಿನ್ನ eps = ((ಭಿನ್ನ *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_back(ggml_ctx, src_clone[0], src_clone[1], eps);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SILU_BACK) {
            tensor_clone = ggml_silu_back(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_L2_NORM) {
            ಧ್ರುವ ಭಿನ್ನ eps = ((ಭಿನ್ನ *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SOFT_MAX) {
            ಯದಿ (tensor->src[1] != nullptr) {
                ಧ್ರುವ ಭಿನ್ನ * params = (ಧ್ರುವ ಭಿನ್ನ *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } ಅನ್ಯಥಾ {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SOFT_MAX_BACK) {
            tensor_clone = ggml_soft_max_ext_back(ggml_ctx, src_clone[0], src_clone[1], ((ಭಿನ್ನ *)tensor->op_params)[0], ((ಭಿನ್ನ *)tensor->op_params)[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_DIAG_MASK_INF) {
            tensor_clone = ggml_diag_mask_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_BACK) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims      = ((int32_t *) tensor->op_params)[1];
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            ಧ್ರುವ ಭಿನ್ನ freq_base       = ((ಭಿನ್ನ *) tensor->op_params)[5];
            ಧ್ರುವ ಭಿನ್ನ freq_scale      = ((ಭಿನ್ನ *) tensor->op_params)[6];
            ಧ್ರುವ ಭಿನ್ನ ext_factor      = ((ಭಿನ್ನ *) tensor->op_params)[7];
            ಧ್ರುವ ಭಿನ್ನ attn_factor     = ((ಭಿನ್ನ *) tensor->op_params)[8];
            ಧ್ರುವ ಭಿನ್ನ beta_fast       = ((ಭಿನ್ನ *) tensor->op_params)[9];
            ಧ್ರುವ ಭಿನ್ನ beta_slow       = ((ಭಿನ್ನ *) tensor->op_params)[10];
            ಯದಿ (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                ಯದಿ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } ಅನ್ಯಥಾ {
                    tensor_clone = ggml_rope_multi_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            } ಅನ್ಯಥಾ {
                ಯದಿ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } ಅನ್ಯಥಾ {
                    tensor_clone = ggml_rope_ext_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            }
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs = ((int32_t *) tensor->op_params)[15];
            ಯದಿ (n_offs != 0) {
                tensor_clone = ggml_rope_set_offset(tensor_clone, n_offs);
            }
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_UNARY) {
            ವಿಕಲ್ಪ (ggml_get_unary_op(tensor)) {
            ಪಕ್ಷ GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_GELU_QUICK:
                tensor_clone = ggml_gelu_quick(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_set_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_set_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_set_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_set_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_STEP:
                tensor_clone = ggml_step(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_ROUND:
                tensor_clone = ggml_round(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
                GGML_ABORT("fatal error");
            }
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_GLU) {
            ಯದಿ (src_clone[1] == nullptr) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } ಅನ್ಯಥಾ {
                tensor_clone = ggml_glu_split(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_set_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_set_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            ಯದಿ (tensor->src[1] == nullptr) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } ಅನ್ಯಥಾ {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_PERMUTE) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_permute(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_GET_ROWS) {
            tensor_clone = ggml_get_rows(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_sort_order) *(ಪೂರ್ಣಾಙ್ಕ *)tensor->op_params);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ತುದಿ_K) {
            tensor_clone = ggml_ತುದಿ_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SUM) {
            tensor_clone = ggml_sum(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SUM_ROWS) {
            tensor_clone = ggml_sum_rows(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_DSV4_HC_PRE) {
            ಯದಿ (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } ಅನ್ಯಥಾ {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_MEAN) {
            tensor_clone = ggml_mean(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_BACK) {
            tensor_clone = ggml_cross_entropy_loss_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ಎಣಿಕೆ_EQUAL) {
            tensor_clone = ggml_ಎಣಿಕೆ_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], ಸತ್ಯ, ಸತ್ಯ, ಅಸತ್ಯ);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_IM2COL) {
            ಧ್ರುವ int32_t s0 = tensor->op_params[0];
            ಧ್ರುವ int32_t s1 = tensor->op_params[1];
            ಧ್ರುವ int32_t p0 = tensor->op_params[2];
            ಧ್ರುವ int32_t p1 = tensor->op_params[3];
            ಧ್ರುವ int32_t d0 = tensor->op_params[4];
            ಧ್ರುವ int32_t d1 = tensor->op_params[5];

            ಧ್ರುವ ತಾರ್ಕಿಕ is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_IM2COL_3D) {
            ಧ್ರುವ int32_t s0 = tensor->op_params[0];
            ಧ್ರುವ int32_t s1 = tensor->op_params[1];
            ಧ್ರುವ int32_t s2 = tensor->op_params[2];
            ಧ್ರುವ int32_t p0 = tensor->op_params[3];
            ಧ್ರುವ int32_t p1 = tensor->op_params[4];
            ಧ್ರುವ int32_t p2 = tensor->op_params[5];
            ಧ್ರುವ int32_t d0 = tensor->op_params[6];
            ಧ್ರುವ int32_t d1 = tensor->op_params[7];
            ಧ್ರುವ int32_t d2 = tensor->op_params[8];
            ಧ್ರುವ int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            ಧ್ರುವ int32_t dim = tensor->op_params[0];
            ಧ್ರುವ int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            ಧ್ರುವ int32_t s0 = tensor->op_params[0];
            ಧ್ರುವ int32_t p0 = tensor->op_params[1];
            ಧ್ರುವ int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_COL2IM_1D) {
            ಧ್ರುವ int32_t stride = tensor->op_params[0];
            ಧ್ರುವ int32_t oc     = tensor->op_params[1];
            ಧ್ರುವ int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_POOL_1D) {
            ಗಣನಾ ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ಧ್ರುವ int32_t k0 = tensor->op_params[1];
            ಧ್ರುವ int32_t s0 = tensor->op_params[2];
            ಧ್ರುವ int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_POOL_2D) {
            ಗಣನಾ ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ಧ್ರುವ int32_t k0 = tensor->op_params[1];
            ಧ್ರುವ int32_t k1 = tensor->op_params[2];
            ಧ್ರುವ int32_t s0 = tensor->op_params[3];
            ಧ್ರುವ int32_t s1 = tensor->op_params[4];
            ಧ್ರುವ int32_t p0 = tensor->op_params[5];
            ಧ್ರುವ int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CONV_2D) {
            ಧ್ರುವ int32_t s0 = tensor->op_params[0];
            ಧ್ರುವ int32_t s1 = tensor->op_params[1];
            ಧ್ರುವ int32_t p0 = tensor->op_params[2];
            ಧ್ರುವ int32_t p1 = tensor->op_params[3];
            ಧ್ರುವ int32_t d0 = tensor->op_params[4];
            ಧ್ರುವ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CONV_3D) {
            ಧ್ರುವ int32_t s0 = tensor->op_params[0];
            ಧ್ರುವ int32_t s1 = tensor->op_params[1];
            ಧ್ರುವ int32_t s2 = tensor->op_params[2];
            ಧ್ರುವ int32_t p0 = tensor->op_params[3];
            ಧ್ರುವ int32_t p1 = tensor->op_params[4];
            ಧ್ರುವ int32_t p2 = tensor->op_params[5];
            ಧ್ರುವ int32_t d0 = tensor->op_params[6];
            ಧ್ರುವ int32_t d1 = tensor->op_params[7];
            ಧ್ರುವ int32_t d2 = tensor->op_params[8];
            ಧ್ರುವ int32_t IC = tensor->op_params[9];
            ಧ್ರುವ int32_t N  = tensor->op_params[10];
            ಧ್ರುವ int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CONV_2D_DW) {
            ಧ್ರುವ int32_t s0 = tensor->op_params[0];
            ಧ್ರುವ int32_t s1 = tensor->op_params[1];
            ಧ್ರುವ int32_t p0 = tensor->op_params[2];
            ಧ್ರುವ int32_t p1 = tensor->op_params[3];
            ಧ್ರುವ int32_t d0 = tensor->op_params[4];
            ಧ್ರುವ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            ಧ್ರುವ int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_LEAKY_RELU) {
            ಧ್ರುವ ಭಿನ್ನ * op_params = (ಧ್ರುವ ಭಿನ್ನ *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], ಅಸತ್ಯ);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_GATED_LINEAR_ATTN) {
            ಧ್ರುವ ಭಿನ್ನ * op_params = (ಧ್ರುವ ಭಿನ್ನ *)tensor->op_params;
            tensor_clone = ggml_gated_linear_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_OPT_STEP_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_OPT_STEP_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ADD_ID) {
            tensor_clone = ggml_add_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SSM_SCAN) {
            ಧ್ರುವ int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_scan(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } ಅನ್ಯಥಾ ಯದಿ (tensor->op == GGML_OP_ROLL) {
            ಧ್ರುವ int32_t s0 = tensor->op_params[0];
            ಧ್ರುವ int32_t s1 = tensor->op_params[1];
            ಧ್ರುವ int32_t s2 = tensor->op_params[2];
            ಧ್ರುವ int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        ಅನ್ಯಥಾ {
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    ಯದಿ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_size = ggml_nbytes(tensor_clone);

    comp_result = malloc(comp_size);
    memcpy(comp_result, tensor_clone->data, comp_size);
    memcpy(comp_nb, tensor_clone->nb, ಪರಿಮಾಣ(size_t) * GGML_MAX_DIMS);

    ಪ್ರತಿ (auto m : cloned_mallocs) {
        ಮುಕ್ತ(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

ಶೂನ್ಯ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->ಅಂಕೆ_additional_fused_ops];
    ಯದಿ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    ಶೂನ್ಯ * tensor_data = tensor->data;

    ಯದಿ (ggml_backend_buffer_is_vk(tensor->buffer)) {
        size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
        uint64_t offset = vk_tensor_offset(tensor) + tensor->view_offs;
        ಯದಿ (offset + tensor_size >= buffer_gpu->size) {
            tensor_size = buffer_gpu->size - offset;
        }

        ggml_vk_buffer_read(buffer_gpu, offset, tensor_data, tensor_size);
    }

    ಭಿನ್ನ first_error_result = -1.0f;
    ಭಿನ್ನ first_error_correct = -1.0f;
    std::array<ಪೂರ್ಣಾಙ್ಕ, 4> first_error = { -1, -1, -1, -1 };
    ದ್ವಿಗುಣ avg_err = 0.0;
    size_t counter = 0;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i3 = 0; i3 < tensor->ne[3]; i3++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i2 = 0; i2 < tensor->ne[2]; i2++) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i1 = 0; i1 < tensor->ne[1]; i1++) {
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < tensor->ne[0]; i0++) {
                    ಧ್ರುವ ತಾರ್ಕಿಕ buffer_size_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_size;
                    ಭಿನ್ನ correct = 0.0f;
                    ಭಿನ್ನ result = 0.0f;

                    ಯದಿ (buffer_size_fit) {
                        ಯದಿ (tensor->type == GGML_TYPE_F32) {
                            correct = *(ಭಿನ್ನ *) ((ವರ್ಣ *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(ಭಿನ್ನ *) ((ವರ್ಣ *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ಅನ್ಯಥಾ ಯದಿ (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((ವರ್ಣ *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((ವರ್ಣ *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } ಅನ್ಯಥಾ ಯದಿ (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((ವರ್ಣ *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((ವರ್ಣ *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } ಅನ್ಯಥಾ ಯದಿ (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((ವರ್ಣ *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int32_t *) ((ವರ್ಣ *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ಅನ್ಯಥಾ ಯದಿ (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((ವರ್ಣ *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int64_t *) ((ವರ್ಣ *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } ಅನ್ಯಥಾ {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::endl;
                        }
                    } ಅನ್ಯಥಾ {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::endl;
                        GGML_ABORT("fatal error");
                    }

                    ಯದಿ ((std::isnan(correct) != std::isnan(result)) || (std::isinf(correct) != std::isinf(result)) || !buffer_size_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::endl;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
                        ಯದಿ (src0 != nullptr) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
                        }
                        ಯದಿ (src1 != nullptr) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
                        }
                        ಯದಿ (src2 != nullptr) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
                        }
                        ಯದಿ (src3 != nullptr) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
                        }
                        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
                        std::cerr << std::endl << "Result:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
                        std::cerr << std::endl << "Correct:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, comp_result, i0, i1, i2, i3);
                        std::cerr << std::endl;
                        std::vector<ಧ್ರುವ ggml_tensor *> done;
                        ggml_vk_print_graph_origin(tensor, done);
                        GGML_ABORT("fatal error");
                    }
                    ಧ್ರುವ ದ್ವಿಗುಣ denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    ಯದಿ (first_error[0] == -1 && std::fabs(correct - result) / denom > 0.5) {
                        first_error[0] = i0;
                        first_error[1] = i1;
                        first_error[2] = i2;
                        first_error[3] = i3;
                        first_error_result = result;
                        first_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    ಯದಿ (!std::isinf(correct) && !std::isinf(result) && !std::isnan(correct) && !std::isnan(result)) {
                        avg_err += std::fabs(correct - result) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    ಯದಿ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        ಯದಿ (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        ಯದಿ (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        ಯದಿ (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        ಯದಿ (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, 5, 5, 0, 0);
        std::cerr << std::endl;
        std::vector<ಧ್ರುವ ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
    }

    ಯದಿ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        ಯದಿ (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        ಯದಿ (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        ಯದಿ (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        ಯದಿ (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl;
        std::vector<ಧ್ರುವ ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
        GGML_ABORT("fatal error");
    } ಅನ್ಯಥಾ {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::endl;
    }

    ಮುಕ್ತ(comp_result);
    comp_result = nullptr;
    comp_size = 0;

    ಯದಿ (ggml_backend_buffer_is_vk(tensor->buffer)) {
        ಮುಕ್ತ(tensor_data);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#endif

