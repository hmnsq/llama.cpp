#અન્તર્ભાવ "ggml-vulkan-common.h"

તાર્કિક vk_memory_logger_enabled = ખોટું;

તાર્કિક vk_perf_logger_enabled = ખોટું;

તાર્કિક vk_perf_logger_concurrent = ખોટું;

તાર્કિક vk_enable_sync_logger = ખોટું;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

શૂન્ય vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
    જો (!vk_memory_logger_enabled) {
        પાછા;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    અચલ તાર્કિક device = તાર્કિક(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    અચલ std::string type = device ? "device" : "host";
    allocations[buf->buffer] = size;
    total_device += device ? size : 0;
    total_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
}

શૂન્ય vk_memory_logger::log_deallocation(vk_buffer_ref buf_ref) {
    જો (buf_ref.expired() || buf_ref.lock()->size == 0 || !vk_memory_logger_enabled) {
        પાછા;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    અચલ તાર્કિક device = તાર્કિક(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::string type = device ? "device" : "host";
    auto it = allocations.find(buf->buffer);
    જો (it != allocations.end()) {
        total_device -= device ? it->second : 0;
        total_host -= device ? 0 : it->second;
        VK_LOG_MEMORY(buf->device->name << ": -" << format_size(it->second) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
        allocations.erase(it);
    } નહીંતર {
        VK_LOG_MEMORY("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->buffer);
    }
}

#ifdef GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

સ્થિર શૂન્ય ggml_vk_print_tensor(અચલ ggml_tensor * tensor, અચલ અક્ષર * name);
#endif

#ifdef GGML_VULKAN_RUN_TESTS
સ્થિર શૂન્ય ggml_vk_print_matrix_area(અચલ શૂન્ય * data, ggml_type type, પૂર્ણાંક ne0, પૂર્ણાંક ne1, પૂર્ણાંક i0, પૂર્ણાંક i1, પૂર્ણાંક i2) {
    જો (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        પાછા;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    માટે (પૂર્ણાંક idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    માટે (પૂર્ણાંક idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        માટે (પૂર્ણાંક idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            જો (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                દશાંશ val;
                જો (type == GGML_TYPE_F32) {
                    val = *((અચલ દશાંશ *) data + i2*ne1*ne0 + idx1*ne0 + idx0);
                } નહીંતર જો (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*((અચલ ggml_fp16_t *) data + i2*ne1*ne0 + idx1*ne0 + idx0));
                } નહીંતર {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } નહીંતર {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

template <typename X_TYPE, typename Y_TYPE>
શૂન્ય ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t અંક_it, પૂર્ણાંક split_k, પૂર્ણાંક shader_size) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << અંક_it << ", " << split_k << ", " << shader_size << ")");
    અચલ size_t x_ne = m * k * batch;
    અચલ size_t y_ne = k * n * batch;
    અચલ size_t d_ne = m * n * batch;

    ggml_type x_type = std::is_same<દશાંશ, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<દશાંશ, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_ચાવી mm_test_ચાવી{x_type, y_type, ખોટું, ખોટું};
    auto mm_test_it = ctx->device->pipeline_matmul.find(mm_test_ચાવી);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.empty());
    auto& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_size >= 0 && shader_size < (પૂર્ણાંક)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_size);
    vk_pipeline p = mm_test_configs[shader_size].aligned ? mm_test_configs[shader_size].aligned : mm_test_configs[shader_size].unaligned;

    અચલ size_t kpad = ggml_vk_align_size(k, mm_test_configs[shader_size].align);

    જો (k != kpad) {
        p = mm_test_configs[shader_size].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_size);
    }

    જો (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, અંક_it);

        જો (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < કદ(દશાંશ) * d_ne * split_k) {
            // Resize buffer
            જો (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, કદ(દશાંશ) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_buffer d_X = ggml_vk_create_buffer_check(ctx->device, કદ(X_TYPE) * x_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_Y = ggml_vk_create_buffer_check(ctx->device, કદ(Y_TYPE) * y_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_D = ggml_vk_create_buffer_check(ctx->device, કદ(દશાંશ) * d_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) malloc(કદ(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) malloc(કદ(Y_TYPE) * y_ne);
    દશાંશ* d = (દશાંશ *) malloc(કદ(દશાંશ) * d_ne);

    માટે (size_t i = 0; i < x_ne; i++) {
        જો (std::is_same<દશાંશ, X_TYPE>()) {
            x[i] = (rand() / (દશાંશ)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } નહીંતર જો (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_to_fp16((rand() / (દશાંશ)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } નહીંતર {
            GGML_ABORT("fatal error");
        }
    }
    માટે (size_t i = 0; i < y_ne; i++) {
        જો (std::is_same<દશાંશ, Y_TYPE>()) {
            y[i] = (rand() / (દશાંશ)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } નહીંતર જો (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_to_fp16((rand() / (દશાંશ)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } નહીંતર {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_buffer_write(d_X, 0, x, કદ(X_TYPE) * k * m * batch);
    ggml_vk_buffer_write(d_Y, 0, y, કદ(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    માટે (size_t i = 0; i < અંક_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1, n,
            k * કદ(Y_TYPE), m * કદ(દશાંશ)
        );
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, સાચું, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();
    દ્વિગુણ time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_buffer_read(d_D, 0, d, કદ(દશાંશ) * d_ne);

    દશાંશ * d_chk = (દશાંશ *) malloc(કદ(દશાંશ) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    જો (std::is_same<દશાંશ, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } નહીંતર જો (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } નહીંતર {
        GGML_ABORT("fatal error");
    }
    જો (std::is_same<દશાંશ, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } નહીંતર જો (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } નહીંતર {
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

    દ્વિગુણ avg_err = 0.0;
    પૂર્ણાંક first_err_n = -1;
    પૂર્ણાંક first_err_m = -1;
    પૂર્ણાંક first_err_b = -1;

    માટે (size_t i = 0; i < m*n*batch; i++) {
        દ્વિગુણ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        જો ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    દ્વિગુણ tflops = 2.0*m*n*k*batch*અંક_it / (time / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time / અંક_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    જો (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        જો (split_k > 1) {
            દશાંશ * split_k_buf = (દશાંશ *) malloc(કદ(દશાંશ) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, કદ(દશાંશ) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            મુક્ત(split_k_buf);
        }
    }

    મુક્ત(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_buffer(d_X);
    ggml_vk_destroy_buffer(d_Y);
    ggml_vk_destroy_buffer(d_D);

    મુક્ત(x);
    મુક્ત(y);
    મુક્ત(d);
}

template શૂન્ય ggml_vk_test_matmul<ggml_fp16_t, દશાંશ>(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t અંક_it, પૂર્ણાંક split_k, પૂર્ણાંક shader_size);

સ્થિર શૂન્ય ggml_vk_print_tensor_area(અચલ ggml_tensor * tensor, પૂર્ણાંક i0, પૂર્ણાંક i1, પૂર્ણાંક i2, પૂર્ણાંક i3) {
    જો (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        પાછા;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    માટે (પૂર્ણાંક idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    માટે (પૂર્ણાંક idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        માટે (પૂર્ણાંક idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            જો (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                દશાંશ val;
                જો (tensor->type == GGML_TYPE_F32) {
                    val = *(દશાંશ *) ((અક્ષર *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } નહીંતર જો (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((અક્ષર *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } નહીંતર {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } નહીંતર {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

સ્થિર શૂન્ય ggml_vk_quantize_data(અચલ દશાંશ * from, શૂન્ય * to, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, to, 0, 1, ne, nullptr);
}

સ્થિર શૂન્ય ggml_vk_dequantize_data(અચલ શૂન્ય * from, દશાંશ * to, size_t ne, ggml_type quant) {
    જો (quant == GGML_TYPE_F32) {
        memcpy(to, from, કદ(દશાંશ) * ne);
        પાછા;
    }

    અચલ auto * tt = ggml_get_type_traits(quant);

    ggml_to_float_t dequant_fn = tt->to_float;

    dequant_fn(from, to, ne);
}

શૂન્ય ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    અચલ size_t x_sz = કદ(દશાંશ) * ne;
    અચલ size_t x_sz_f16 = કદ(ggml_fp16_t) * ne;
    અચલ size_t qx_sz = ne * ggml_type_size(quant)/ggml_blck_size(quant);
    દશાંશ * x = (દશાંશ *) malloc(x_sz);
    શૂન્ય * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer x_buf = ggml_vk_create_buffer_check(ctx->device, x_sz_f16, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    દશાંશ * x_ref = (દશાંશ *) malloc(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) malloc(x_sz_f16);

    માટે (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (દશાંશ)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_to_fp16(ctx, quant);

    ggml_vk_quantize_data(x, qx, ne, quant);
    ggml_vk_dequantize_data(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    અચલ std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, સાચું, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    દ્વિગુણ ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(x_buf, 0, x_chk, x_sz_f16);

    પૂર્ણાંક first_err = -1;

    દ્વિગુણ avg_err = 0.0;
    માટે (size_t i = 0; i < ne; i++) {
        દ્વિગુણ error = std::fabs(x_ref[i] - ggml_fp16_to_fp32(x_chk[i]));
        avg_err += error;

        જો (first_err < 0 && error > 0.05) {
            first_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::endl;

    જો (avg_err > 0.1) {
        std::cerr << "first_error = " << first_err << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        માટે (પૂર્ણાંક i = std::max(0, first_err - 5); i < std::min((પૂર્ણાંક)ne, first_err + 5); i++) {
            std::cerr << ggml_fp16_to_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
        માટે (પૂર્ણાંક i = std::max(0, first_err - 5); i < std::min((પૂર્ણાંક)ne, first_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::endl;
    }

    ggml_vk_destroy_buffer(x_buf);
    ggml_vk_destroy_buffer(qx_buf);

    મુક્ત(x);
    મુક્ત(qx);
    મુક્ત(x_ref);
    મુક્ત(x_chk);
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

શૂન્ય ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t અંક_it, size_t split_k, size_t shader_size, ggml_type quant, તાર્કિક mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << અંક_it << ", " << split_k << ", " << ggml_type_name(quant) << ")");
    અચલ size_t x_ne = m * k * batch;
    અચલ size_t y_ne = k * n * batch;
    અચલ size_t d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    તાર્કિક f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_ચાવી dq_ચાવી{quant, b_type, ખોટું, f16acc};
    auto dq_it = ctx->device->pipeline_matmul.find(dq_ચાવી);
    જો (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        જો (f16acc) {
            dq_ચાવી.f16acc = ખોટું;
            dq_it = ctx->device->pipeline_matmul.find(dq_ચાવી);
        }
    }
    જો (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::endl;
        પાછા;
    }
    auto& dq_configs = dq_it->second;
    જો (shader_size >= (પૂર્ણાંક)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_size << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::endl;
        પાછા;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_size);
    vk_pipeline p = dq_configs[shader_size].aligned ? dq_configs[shader_size].aligned : dq_configs[shader_size].unaligned;

    અચલ size_t kpad = mmq ? 0 : ggml_vk_align_size(k, dq_configs[shader_size].align);

    જો (mmq || k != kpad) {
        p = dq_configs[shader_size].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_size);
    }

    જો (p == nullptr) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::endl;
        પાછા;
    }

    અચલ size_t x_sz = કદ(દશાંશ) * x_ne;
    અચલ size_t y_sz = કદ(દશાંશ) * y_ne;
    અચલ size_t qx_sz = x_ne * ggml_type_size(quant)/ggml_blck_size(quant);
    અચલ size_t qy_sz = mmq ? y_ne * ggml_type_size(GGML_TYPE_Q8_1)/ggml_blck_size(GGML_TYPE_Q8_1) : y_sz;
    અચલ size_t d_sz = કદ(દશાંશ) * d_ne;
    દશાંશ * x = (દશાંશ *) malloc(x_sz);
    દશાંશ * y = (દશાંશ *) malloc(y_sz);
    શૂન્ય * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer y_buf = ggml_vk_create_buffer_check(ctx->device, y_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer qy_buf = ggml_vk_create_buffer_check(ctx->device, qy_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_buf = ggml_vk_create_buffer_check(ctx->device, d_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    દશાંશ * d = (દશાંશ *) malloc(d_sz);
    દશાંશ * d_chk = (દશાંશ *) malloc(d_sz);

    માટે (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (દશાંશ)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_data(x, qx, x_ne, quant);

    માટે (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (દશાંશ)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    જો (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, અંક_it);

        જો (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < કદ(દશાંશ) * d_ne * split_k) {
            // Resize buffer
            જો (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, કદ(દશાંશ) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }
    જો (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, અંક_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_buffer_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    જો (mmq) {
        માટે (size_t i = 0; i < અંક_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_size(GGML_TYPE_Q8_1)) * ggml_type_size(GGML_TYPE_Q8_1), m * કદ(દશાંશ)
            );
        }
    } નહીંતર {
        માટે (size_t i = 0; i < અંક_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                k * કદ(દશાંશ), m * કદ(દશાંશ)
            );
        }
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, સાચું, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    દ્વિગુણ time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
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

    દ્વિગુણ avg_err = 0.0;
    પૂર્ણાંક first_err_n = -1;
    પૂર્ણાંક first_err_m = -1;
    પૂર્ણાંક first_err_b = -1;

    માટે (size_t i = 0; i < m*n*batch; i++) {
        દ્વિગુણ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        જો ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    દ્વિગુણ tflops = 2.0*m*n*k*batch*અંક_it / (time_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    જો (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time_ms / અંક_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    જો (avg_err > 0.01 || std::isnan(avg_err)) {
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

        જો (split_k > 1) {
            દશાંશ * split_k_buf = (દશાંશ *) malloc(કદ(દશાંશ) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, કદ(દશાંશ) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            મુક્ત(split_k_buf);
        }
    }

    ggml_vk_destroy_buffer(qx_buf);
    ggml_vk_destroy_buffer(y_buf);
    ggml_vk_destroy_buffer(qy_buf);
    ggml_vk_destroy_buffer(d_buf);

    મુક્ત(x);
    મુક્ત(qx);
    મુક્ત(y);
    મુક્ત(d);
    મુક્ત(d_chk);
}
#endif

int64_t ggml_vk_get_op_batch_size(અચલ ggml_tensor * op) {
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_GET_ROWS:
            પાછા 0;
        સ્થિતિ GGML_OP_MUL_MAT:
            પાછા op->ne[1];
        સ્થિતિ GGML_OP_MUL_MAT_ID:
        સ્થિતિ GGML_OP_ROPE:
        સ્થિતિ GGML_OP_ROPE_BACK:
            પાછા op->ne[2];
        મૂળભૂત:
            પાછા ggml_nrows(op);
    }
}

#ifdef GGML_VULKAN_CHECK_RESULTS
સ્થિર શૂન્ય ggml_vk_print_graph_origin(અચલ ggml_tensor * tensor, std::vector<અચલ ggml_tensor *>& done, પૂર્ણાંક level = 0) {
    જો (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) {
        પાછા;
    }
    માટે (પૂર્ણાંક j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;

    done.push_back(tensor);

    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        જો (tensor->src[i] != nullptr) {
            ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

સ્થિર શૂન્ય ggml_vk_print_tensor_area(અચલ ggml_tensor * tensor, અચલ શૂન્ય * data, પૂર્ણાંક i0, પૂર્ણાંક i1, પૂર્ણાંક i2, પૂર્ણાંક i3) {
    જો (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        પાછા;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    માટે (પૂર્ણાંક idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    માટે (પૂર્ણાંક idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        માટે (પૂર્ણાંક idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            જો (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                દશાંશ val;
                જો (tensor->type == GGML_TYPE_F32) {
                    val = *(અચલ દશાંશ *) ((અચલ અક્ષર *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } નહીંતર જો (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(અચલ ggml_fp16_t *) ((અચલ અક્ષર *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } નહીંતર જો (tensor->type == GGML_TYPE_I32) {
                    val = *(અચલ int32_t *) ((અચલ અક્ષર *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } નહીંતર {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } નહીંતર {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

સ્થિર શૂન્ય ggml_vk_print_tensor(અચલ ggml_tensor * tensor, અચલ અક્ષર * name) {
    શૂન્ય * tensor_data = tensor->data;

    અચલ તાર્કિક is_gpu = tensor->buffer != nullptr && ggml_backend_buffer_is_vk(tensor->buffer);

    જો (is_gpu) {
        અચલ size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer buffer_gpu = buf_ctx->dev_buffer;
        ggml_vk_buffer_read(buffer_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_data, tensor_size);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::endl;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
    જો (tensor->src[0] != nullptr) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::endl;
    }
    જો (tensor->src[1] != nullptr) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::endl;
    }
    std::cerr << std::endl << "Result:" << std::endl;
    ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<અચલ ggml_tensor *> done;
    ggml_vk_print_graph_origin(tensor, done);

    જો (is_gpu) {
        મુક્ત(tensor_data);
    }
}

શૂન્ય * comp_result;
size_t comp_size;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
શૂન્ય ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, પૂર્ણાંક tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->અંક_additional_fused_ops];
    જો (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        પાછા;
    }

    check_counter++;
    જો (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        પાછા;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    સંરચના ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ ખોટું,
    };

    સંરચના ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<સંરચના ggml_tensor *, GGML_MAX_SRC> src_clone = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    અચલ અક્ષર * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<શૂન્ય *> cloned_mallocs;

    સંરચના ggml_tensor * tensor_clone = nullptr;

    માટે (પૂર્ણાંક f = 0; f < ctx->અંક_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            જો (srci == nullptr) {
                ચાલુ;
            }
            // If a src tensor has been cloned, use that one
            auto it = cloned_tensors.find(srci);
            જો (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                ચાલુ;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            size_t srci_size = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            શૂન્ય *src_buffer = malloc(srci_size);
            cloned_mallocs.push_back(src_buffer);

            srci_clone->data = src_buffer;
            જો (ggml_backend_buffer_is_host(srci->buffer)) {
                memcpy(srci_clone->data, srci->data, srci_size);
                memcpy(srci_clone->nb, srci->nb, કદ(size_t) * GGML_MAX_DIMS);
            } નહીંતર જો (ggml_backend_buffer_is_vk(srci->buffer)) {
                ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)srci->buffer->context;
                vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
                uint64_t offset = vk_tensor_offset(srci) + srci->view_offs;
                જો (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    માટે (પૂર્ણાંક i3 = 0; i3 < srci->ne[3]; i3++) {
                        માટે (પૂર્ણાંક i2 = 0; i2 < srci->ne[2]; i2++) {
                            અચલ પૂર્ણાંક idx = i3*srci->ne[2] + i2;
                            ggml_vk_buffer_read(buffer_gpu, offset + idx * srci->nb[2], ((અક્ષર *)srci_clone->data + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    માટે (પૂર્ણાંક i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } નહીંતર {
                    જો (offset + srci_size >= buffer_gpu->size) {
                        srci_size = buffer_gpu->size - offset;
                    }
                    ggml_vk_buffer_read(buffer_gpu, offset, srci_clone->data, srci_size);
                    memcpy(srci_clone->nb, srci->nb, કદ(size_t) * GGML_MAX_DIMS);
                }
            } નહીંતર {
                GGML_ABORT("fatal error");
            }

            જો (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        જો (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            અચલ દશાંશ * params = (અચલ દશાંશ *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            જો (src_clone[4]) {
                ggml_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]);
            }
        } નહીંતર જો (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } નહીંતર જો (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(પૂર્ણાંક *)tensor->op_params);
        } નહીંતર જો (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_mode) tensor->op_params[0]);
        } નહીંતર જો (tensor->op == GGML_OP_SCALE) {
            અચલ દશાંશ * params = (અચલ દશાંશ *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } નહીંતર જો (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_ARANGE) {
            અચલ દશાંશ start = ggml_get_op_params_f32(tensor, 0);
            અચલ દશાંશ stop = ggml_get_op_params_f32(tensor, 1);
            અચલ દશાંશ step = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, step);
        } નહીંતર જો (tensor->op == GGML_OP_FILL) {
            અચલ દશાંશ value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } નહીંતર જો (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } નહીંતર જો (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_CLAMP) {
            અચલ દશાંશ * params = (અચલ દશાંશ *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } નહીંતર જો (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } નહીંતર જો (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } નહીંતર જો (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } નહીંતર જો (tensor->op == GGML_OP_REPEAT_BACK) {
            tensor_clone = ggml_repeat_back(ggml_ctx, src_clone[0], tensor);
        } નહીંતર જો (tensor->op == GGML_OP_ADD) {
            tensor_clone = ggml_add(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } નહીંતર જો (tensor->op == GGML_OP_SET) {
            tensor_clone = ggml_set(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } નહીંતર જો (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(દશાંશ *)tensor->op_params);
        } નહીંતર જો (tensor->op == GGML_OP_GROUP_NORM) {
            અચલ દશાંશ * float_params = (અચલ દશાંશ *)tensor->op_params;
            tensor_clone = ggml_group_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } નહીંતર જો (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(દશાંશ *)tensor->op_params);
        } નહીંતર જો (tensor->op == GGML_OP_RMS_NORM_BACK) {
            અચલ દશાંશ eps = ((દશાંશ *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_back(ggml_ctx, src_clone[0], src_clone[1], eps);
        } નહીંતર જો (tensor->op == GGML_OP_SILU_BACK) {
            tensor_clone = ggml_silu_back(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_L2_NORM) {
            અચલ દશાંશ eps = ((દશાંશ *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } નહીંતર જો (tensor->op == GGML_OP_SOFT_MAX) {
            જો (tensor->src[1] != nullptr) {
                અચલ દશાંશ * params = (અચલ દશાંશ *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } નહીંતર {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } નહીંતર જો (tensor->op == GGML_OP_SOFT_MAX_BACK) {
            tensor_clone = ggml_soft_max_ext_back(ggml_ctx, src_clone[0], src_clone[1], ((દશાંશ *)tensor->op_params)[0], ((દશાંશ *)tensor->op_params)[1]);
        } નહીંતર જો (tensor->op == GGML_OP_DIAG_MASK_INF) {
            tensor_clone = ggml_diag_mask_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } નહીંતર જો (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_BACK) {
            અચલ પૂર્ણાંક n_dims      = ((int32_t *) tensor->op_params)[1];
            અચલ પૂર્ણાંક mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            અચલ પૂર્ણાંક n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            અચલ દશાંશ freq_base       = ((દશાંશ *) tensor->op_params)[5];
            અચલ દશાંશ freq_scale      = ((દશાંશ *) tensor->op_params)[6];
            અચલ દશાંશ ext_factor      = ((દશાંશ *) tensor->op_params)[7];
            અચલ દશાંશ attn_factor     = ((દશાંશ *) tensor->op_params)[8];
            અચલ દશાંશ beta_fast       = ((દશાંશ *) tensor->op_params)[9];
            અચલ દશાંશ beta_slow       = ((દશાંશ *) tensor->op_params)[10];
            જો (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                જો (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } નહીંતર {
                    tensor_clone = ggml_rope_multi_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            } નહીંતર {
                જો (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } નહીંતર {
                    tensor_clone = ggml_rope_ext_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            }
            અચલ પૂર્ણાંક n_offs = ((int32_t *) tensor->op_params)[15];
            જો (n_offs != 0) {
                tensor_clone = ggml_rope_set_offset(tensor_clone, n_offs);
            }
        } નહીંતર જો (tensor->op == GGML_OP_UNARY) {
            પસંદગી (ggml_get_unary_op(tensor)) {
            સ્થિતિ GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_GELU_QUICK:
                tensor_clone = ggml_gelu_quick(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_set_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_set_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_set_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_set_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                તોડો;
            સ્થિતિ GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_STEP:
                tensor_clone = ggml_step(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_ROUND:
                tensor_clone = ggml_round(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                તોડો;
            સ્થિતિ GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                તોડો;
            મૂળભૂત:
                std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
                GGML_ABORT("fatal error");
            }
        } નહીંતર જો (tensor->op == GGML_OP_GLU) {
            જો (src_clone[1] == nullptr) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } નહીંતર {
                tensor_clone = ggml_glu_split(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_set_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_set_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } નહીંતર જો (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            જો (tensor->src[1] == nullptr) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } નહીંતર {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } નહીંતર જો (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } નહીંતર જો (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } નહીંતર જો (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } નહીંતર જો (tensor->op == GGML_OP_PERMUTE) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_permute(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } નહીંતર જો (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_GET_ROWS) {
            tensor_clone = ggml_get_rows(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_sort_order) *(પૂર્ણાંક *)tensor->op_params);
        } નહીંતર જો (tensor->op == GGML_OP_ટોચ_K) {
            tensor_clone = ggml_ટોચ_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } નહીંતર જો (tensor->op == GGML_OP_SUM) {
            tensor_clone = ggml_sum(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_SUM_ROWS) {
            tensor_clone = ggml_sum_rows(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } નહીંતર જો (tensor->op == GGML_OP_DSV4_HC_PRE) {
            જો (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } નહીંતર {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } નહીંતર જો (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } નહીંતર જો (tensor->op == GGML_OP_MEAN) {
            tensor_clone = ggml_mean(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } નહીંતર જો (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_BACK) {
            tensor_clone = ggml_cross_entropy_loss_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } નહીંતર જો (tensor->op == GGML_OP_ગણતરી_EQUAL) {
            tensor_clone = ggml_ગણતરી_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], સાચું, સાચું, ખોટું);
        } નહીંતર જો (tensor->op == GGML_OP_IM2COL) {
            અચલ int32_t s0 = tensor->op_params[0];
            અચલ int32_t s1 = tensor->op_params[1];
            અચલ int32_t p0 = tensor->op_params[2];
            અચલ int32_t p1 = tensor->op_params[3];
            અચલ int32_t d0 = tensor->op_params[4];
            અચલ int32_t d1 = tensor->op_params[5];

            અચલ તાર્કિક is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } નહીંતર જો (tensor->op == GGML_OP_IM2COL_3D) {
            અચલ int32_t s0 = tensor->op_params[0];
            અચલ int32_t s1 = tensor->op_params[1];
            અચલ int32_t s2 = tensor->op_params[2];
            અચલ int32_t p0 = tensor->op_params[3];
            અચલ int32_t p1 = tensor->op_params[4];
            અચલ int32_t p2 = tensor->op_params[5];
            અચલ int32_t d0 = tensor->op_params[6];
            અચલ int32_t d1 = tensor->op_params[7];
            અચલ int32_t d2 = tensor->op_params[8];
            અચલ int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } નહીંતર જો (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            અચલ int32_t dim = tensor->op_params[0];
            અચલ int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } નહીંતર જો (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            અચલ int32_t s0 = tensor->op_params[0];
            અચલ int32_t p0 = tensor->op_params[1];
            અચલ int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } નહીંતર જો (tensor->op == GGML_OP_COL2IM_1D) {
            અચલ int32_t stride = tensor->op_params[0];
            અચલ int32_t oc     = tensor->op_params[1];
            અચલ int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } નહીંતર જો (tensor->op == GGML_OP_POOL_1D) {
            ગણના ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            અચલ int32_t k0 = tensor->op_params[1];
            અચલ int32_t s0 = tensor->op_params[2];
            અચલ int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } નહીંતર જો (tensor->op == GGML_OP_POOL_2D) {
            ગણના ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            અચલ int32_t k0 = tensor->op_params[1];
            અચલ int32_t k1 = tensor->op_params[2];
            અચલ int32_t s0 = tensor->op_params[3];
            અચલ int32_t s1 = tensor->op_params[4];
            અચલ int32_t p0 = tensor->op_params[5];
            અચલ int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } નહીંતર જો (tensor->op == GGML_OP_CONV_2D) {
            અચલ int32_t s0 = tensor->op_params[0];
            અચલ int32_t s1 = tensor->op_params[1];
            અચલ int32_t p0 = tensor->op_params[2];
            અચલ int32_t p1 = tensor->op_params[3];
            અચલ int32_t d0 = tensor->op_params[4];
            અચલ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } નહીંતર જો (tensor->op == GGML_OP_CONV_3D) {
            અચલ int32_t s0 = tensor->op_params[0];
            અચલ int32_t s1 = tensor->op_params[1];
            અચલ int32_t s2 = tensor->op_params[2];
            અચલ int32_t p0 = tensor->op_params[3];
            અચલ int32_t p1 = tensor->op_params[4];
            અચલ int32_t p2 = tensor->op_params[5];
            અચલ int32_t d0 = tensor->op_params[6];
            અચલ int32_t d1 = tensor->op_params[7];
            અચલ int32_t d2 = tensor->op_params[8];
            અચલ int32_t IC = tensor->op_params[9];
            અચલ int32_t N  = tensor->op_params[10];
            અચલ int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } નહીંતર જો (tensor->op == GGML_OP_CONV_2D_DW) {
            અચલ int32_t s0 = tensor->op_params[0];
            અચલ int32_t s1 = tensor->op_params[1];
            અચલ int32_t p0 = tensor->op_params[2];
            અચલ int32_t p1 = tensor->op_params[3];
            અચલ int32_t d0 = tensor->op_params[4];
            અચલ int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } નહીંતર જો (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            અચલ int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } નહીંતર જો (tensor->op == GGML_OP_LEAKY_RELU) {
            અચલ દશાંશ * op_params = (અચલ દશાંશ *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], ખોટું);
        } નહીંતર જો (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } નહીંતર જો (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } નહીંતર જો (tensor->op == GGML_OP_GATED_LINEAR_ATTN) {
            અચલ દશાંશ * op_params = (અચલ દશાંશ *)tensor->op_params;
            tensor_clone = ggml_gated_linear_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } નહીંતર જો (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } નહીંતર જો (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } નહીંતર જો (tensor->op == GGML_OP_OPT_STEP_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } નહીંતર જો (tensor->op == GGML_OP_OPT_STEP_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } નહીંતર જો (tensor->op == GGML_OP_ADD_ID) {
            tensor_clone = ggml_add_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } નહીંતર જો (tensor->op == GGML_OP_SSM_SCAN) {
            અચલ int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_scan(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } નહીંતર જો (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } નહીંતર જો (tensor->op == GGML_OP_ROLL) {
            અચલ int32_t s0 = tensor->op_params[0];
            અચલ int32_t s1 = tensor->op_params[1];
            અચલ int32_t s2 = tensor->op_params[2];
            અચલ int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        નહીંતર {
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    જો (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_size = ggml_nbytes(tensor_clone);

    comp_result = malloc(comp_size);
    memcpy(comp_result, tensor_clone->data, comp_size);
    memcpy(comp_nb, tensor_clone->nb, કદ(size_t) * GGML_MAX_DIMS);

    માટે (auto m : cloned_mallocs) {
        મુક્ત(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

શૂન્ય ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, પૂર્ણાંક tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->અંક_additional_fused_ops];
    જો (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        પાછા;
    }

    જો (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        પાછા;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    શૂન્ય * tensor_data = tensor->data;

    જો (ggml_backend_buffer_is_vk(tensor->buffer)) {
        size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
        uint64_t offset = vk_tensor_offset(tensor) + tensor->view_offs;
        જો (offset + tensor_size >= buffer_gpu->size) {
            tensor_size = buffer_gpu->size - offset;
        }

        ggml_vk_buffer_read(buffer_gpu, offset, tensor_data, tensor_size);
    }

    દશાંશ first_error_result = -1.0f;
    દશાંશ first_error_correct = -1.0f;
    std::array<પૂર્ણાંક, 4> first_error = { -1, -1, -1, -1 };
    દ્વિગુણ avg_err = 0.0;
    size_t counter = 0;

    માટે (પૂર્ણાંક i3 = 0; i3 < tensor->ne[3]; i3++) {
        માટે (પૂર્ણાંક i2 = 0; i2 < tensor->ne[2]; i2++) {
            માટે (પૂર્ણાંક i1 = 0; i1 < tensor->ne[1]; i1++) {
                માટે (પૂર્ણાંક i0 = 0; i0 < tensor->ne[0]; i0++) {
                    અચલ તાર્કિક buffer_size_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_size;
                    દશાંશ correct = 0.0f;
                    દશાંશ result = 0.0f;

                    જો (buffer_size_fit) {
                        જો (tensor->type == GGML_TYPE_F32) {
                            correct = *(દશાંશ *) ((અક્ષર *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(દશાંશ *) ((અક્ષર *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } નહીંતર જો (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((અક્ષર *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((અક્ષર *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } નહીંતર જો (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((અક્ષર *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((અક્ષર *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } નહીંતર જો (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((અક્ષર *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int32_t *) ((અક્ષર *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } નહીંતર જો (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((અક્ષર *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int64_t *) ((અક્ષર *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } નહીંતર {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::endl;
                        }
                    } નહીંતર {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::endl;
                        GGML_ABORT("fatal error");
                    }

                    જો ((std::isnan(correct) != std::isnan(result)) || (std::isinf(correct) != std::isinf(result)) || !buffer_size_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::endl;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
                        જો (src0 != nullptr) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
                        }
                        જો (src1 != nullptr) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
                        }
                        જો (src2 != nullptr) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
                        }
                        જો (src3 != nullptr) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
                        }
                        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
                        std::cerr << std::endl << "Result:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
                        std::cerr << std::endl << "Correct:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, comp_result, i0, i1, i2, i3);
                        std::cerr << std::endl;
                        std::vector<અચલ ggml_tensor *> done;
                        ggml_vk_print_graph_origin(tensor, done);
                        GGML_ABORT("fatal error");
                    }
                    અચલ દ્વિગુણ denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    જો (first_error[0] == -1 && std::fabs(correct - result) / denom > 0.5) {
                        first_error[0] = i0;
                        first_error[1] = i1;
                        first_error[2] = i2;
                        first_error[3] = i3;
                        first_error_result = result;
                        first_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    જો (!std::isinf(correct) && !std::isinf(result) && !std::isnan(correct) && !std::isnan(result)) {
                        avg_err += std::fabs(correct - result) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    જો (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        જો (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        જો (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        જો (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        જો (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, 5, 5, 0, 0);
        std::cerr << std::endl;
        std::vector<અચલ ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
    }

    જો (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        જો (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        જો (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        જો (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        જો (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl;
        std::vector<અચલ ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
        GGML_ABORT("fatal error");
    } નહીંતર {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::endl;
    }

    મુક્ત(comp_result);
    comp_result = nullptr;
    comp_size = 0;

    જો (ggml_backend_buffer_is_vk(tensor->buffer)) {
        મુક્ત(tensor_data);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#endif

