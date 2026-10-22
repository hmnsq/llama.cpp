#অন্তর্ভাব "ggml-vulkan-common.h"

যৌক্তিক vk_memory_logger_enabled = মিথ্যা;

যৌক্তিক vk_perf_logger_enabled = মিথ্যা;

যৌক্তিক vk_perf_logger_concurrent = মিথ্যা;

যৌক্তিক vk_enable_sync_logger = মিথ্যা;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

শূন্য vk_memory_logger::log_allocation(vk_বাফার_ref buf_ref, size_t size) {
    যদি (!vk_memory_logger_enabled) {
        ফেরত;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_বাফার buf = buf_ref.lock();
    ধ্রুবক যৌক্তিক device = যৌক্তিক(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    ধ্রুবক std::string type = device ? "device" : "host";
    allocations[buf->বাফার] = size;
    যোগ_device += device ? size : 0;
    যোগ_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_আকার(size) << " " << type << " at " << buf->বাফার << ". Total device: " << format_আকার(যোগ_device) << ", total host: " << format_আকার(যোগ_host));
}

শূন্য vk_memory_logger::log_deallocation(vk_বাফার_ref buf_ref) {
    যদি (buf_ref.expired() || buf_ref.lock()->size == 0 || !vk_memory_logger_enabled) {
        ফেরত;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_বাফার buf = buf_ref.lock();
    ধ্রুবক যৌক্তিক device = যৌক্তিক(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::string type = device ? "device" : "host";
    auto it = allocations.find(buf->বাফার);
    যদি (it != allocations.end()) {
        যোগ_device -= device ? it->second : 0;
        যোগ_host -= device ? 0 : it->second;
        VK_LOG_MEMORY(buf->device->name << ": -" << format_আকার(it->second) << " " << type << " at " << buf->বাফার << ". Total device: " << format_আকার(যোগ_device) << ", total host: " << format_আকার(যোগ_host));
        allocations.erase(it);
    } অন্যথা {
        VK_LOG_MEMORY("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->বাফার);
    }
}

#ifdef GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

স্থির শূন্য ggml_vk_print_tensor(ধ্রুবক ggml_tensor * tensor, ধ্রুবক অক্ষর * name);
#endif

#ifdef GGML_VULKAN_RUN_TESTS
স্থির শূন্য ggml_vk_print_matrix_area(ধ্রুবক শূন্য * তথ্য, ggml_type type, পূর্ণসংখ্যা ne0, পূর্ণসংখ্যা ne1, পূর্ণসংখ্যা i0, পূর্ণসংখ্যা i1, পূর্ণসংখ্যা i2) {
    যদি (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        ফেরত;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    জন্য (পূর্ণসংখ্যা idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    জন্য (পূর্ণসংখ্যা idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        জন্য (পূর্ণসংখ্যা idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            যদি (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                দশমিক val;
                যদি (type == GGML_TYPE_F32) {
                    val = *((ধ্রুবক দশমিক *) তথ্য + i2*ne1*ne0 + idx1*ne0 + idx0);
                } অন্যথা যদি (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*((ধ্রুবক ggml_fp16_t *) তথ্য + i2*ne1*ne0 + idx1*ne0 + idx0));
                } অন্যথা {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } অন্যথা {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

template <typename X_TYPE, typename Y_TYPE>
শূন্য ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t অঙ্ক_it, পূর্ণসংখ্যা split_k, পূর্ণসংখ্যা shader_আকার) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << অঙ্ক_it << ", " << split_k << ", " << shader_আকার << ")");
    ধ্রুবক size_t x_ne = m * k * batch;
    ধ্রুবক size_t y_ne = k * n * batch;
    ধ্রুবক size_t d_ne = m * n * batch;

    ggml_type x_type = std::is_same<দশমিক, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<দশমিক, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_চাবি mm_test_চাবি{x_type, y_type, মিথ্যা, মিথ্যা};
    auto mm_test_it = ctx->device->pipeline_matmul.find(mm_test_চাবি);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.empty());
    auto& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_আকার >= 0 && shader_আকার < (পূর্ণসংখ্যা)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_আকার);
    vk_pipeline p = mm_test_configs[shader_আকার].aligned ? mm_test_configs[shader_আকার].aligned : mm_test_configs[shader_আকার].unaligned;

    ধ্রুবক size_t kpad = ggml_vk_align_আকার(k, mm_test_configs[shader_আকার].align);

    যদি (k != kpad) {
        p = mm_test_configs[shader_আকার].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_আকার);
    }

    যদি (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, অঙ্ক_it);

        যদি (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < আকার(দশমিক) * d_ne * split_k) {
            // Resize buffer
            যদি (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_বাফার(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_বাফার_check(ctx->device, আকার(দশমিক) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_বাফার d_X = ggml_vk_create_বাফার_check(ctx->device, আকার(X_TYPE) * x_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_বাফার d_Y = ggml_vk_create_বাফার_check(ctx->device, আকার(Y_TYPE) * y_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_বাফার d_D = ggml_vk_create_বাফার_check(ctx->device, আকার(দশমিক) * d_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) malloc(আকার(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) malloc(আকার(Y_TYPE) * y_ne);
    দশমিক* d = (দশমিক *) malloc(আকার(দশমিক) * d_ne);

    জন্য (size_t i = 0; i < x_ne; i++) {
        যদি (std::is_same<দশমিক, X_TYPE>()) {
            x[i] = (rand() / (দশমিক)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } অন্যথা যদি (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_to_fp16((rand() / (দশমিক)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } অন্যথা {
            GGML_ABORT("fatal error");
        }
    }
    জন্য (size_t i = 0; i < y_ne; i++) {
        যদি (std::is_same<দশমিক, Y_TYPE>()) {
            y[i] = (rand() / (দশমিক)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } অন্যথা যদি (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_to_fp16((rand() / (দশমিক)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } অন্যথা {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_বাফার_write(d_X, 0, x, আকার(X_TYPE) * k * m * batch);
    ggml_vk_বাফার_write(d_Y, 0, y, আকার(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    জন্য (size_t i = 0; i < অঙ্ক_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1, n,
            k * আকার(Y_TYPE), m * আকার(দশমিক)
        );
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, সত্য, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();
    দ্বিগুণ time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_বাফার_read(d_D, 0, d, আকার(দশমিক) * d_ne);

    দশমিক * d_chk = (দশমিক *) malloc(আকার(দশমিক) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    যদি (std::is_same<দশমিক, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } অন্যথা যদি (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } অন্যথা {
        GGML_ABORT("fatal error");
    }
    যদি (std::is_same<দশমিক, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } অন্যথা যদি (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } অন্যথা {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->তথ্য = x;
    src1_ggml->তথ্য = y;
    tensor_ggml->তথ্য = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    দ্বিগুণ avg_err = 0.0;
    পূর্ণসংখ্যা first_err_n = -1;
    পূর্ণসংখ্যা first_err_m = -1;
    পূর্ণসংখ্যা first_err_b = -1;

    জন্য (size_t i = 0; i < m*n*batch; i++) {
        দ্বিগুণ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        যদি ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    দ্বিগুণ tflops = 2.0*m*n*k*batch*অঙ্ক_it / (time / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time / অঙ্ক_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    যদি (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        যদি (split_k > 1) {
            দশমিক * split_k_buf = (দশমিক *) malloc(আকার(দশমিক) * d_ne * split_k);
            ggml_vk_বাফার_read(ctx->prealloc_split_k, 0, split_k_buf, আকার(দশমিক) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            মুক্ত(split_k_buf);
        }
    }

    মুক্ত(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_বাফার(d_X);
    ggml_vk_destroy_বাফার(d_Y);
    ggml_vk_destroy_বাফার(d_D);

    মুক্ত(x);
    মুক্ত(y);
    মুক্ত(d);
}

template শূন্য ggml_vk_test_matmul<ggml_fp16_t, দশমিক>(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t অঙ্ক_it, পূর্ণসংখ্যা split_k, পূর্ণসংখ্যা shader_আকার);

স্থির শূন্য ggml_vk_print_tensor_area(ধ্রুবক ggml_tensor * tensor, পূর্ণসংখ্যা i0, পূর্ণসংখ্যা i1, পূর্ণসংখ্যা i2, পূর্ণসংখ্যা i3) {
    যদি (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        ফেরত;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    জন্য (পূর্ণসংখ্যা idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    জন্য (পূর্ণসংখ্যা idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        জন্য (পূর্ণসংখ্যা idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            যদি (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                দশমিক val;
                যদি (tensor->type == GGML_TYPE_F32) {
                    val = *(দশমিক *) ((অক্ষর *) tensor->তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } অন্যথা যদি (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((অক্ষর *) tensor->তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } অন্যথা {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } অন্যথা {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

স্থির শূন্য ggml_vk_quantize_তথ্য(ধ্রুবক দশমিক * from, শূন্য * to, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, to, 0, 1, ne, nullptr);
}

স্থির শূন্য ggml_vk_dequantize_তথ্য(ধ্রুবক শূন্য * from, দশমিক * to, size_t ne, ggml_type quant) {
    যদি (quant == GGML_TYPE_F32) {
        memcpy(to, from, আকার(দশমিক) * ne);
        ফেরত;
    }

    ধ্রুবক auto * tt = ggml_get_type_traits(quant);

    ggml_to_float_t dequant_fn = tt->to_float;

    dequant_fn(from, to, ne);
}

শূন্য ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    ধ্রুবক size_t x_sz = আকার(দশমিক) * ne;
    ধ্রুবক size_t x_sz_f16 = আকার(ggml_fp16_t) * ne;
    ধ্রুবক size_t qx_sz = ne * ggml_type_আকার(quant)/ggml_blck_আকার(quant);
    দশমিক * x = (দশমিক *) malloc(x_sz);
    শূন্য * qx = malloc(qx_sz);
    vk_বাফার qx_buf = ggml_vk_create_বাফার_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_বাফার x_buf = ggml_vk_create_বাফার_check(ctx->device, x_sz_f16, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    দশমিক * x_ref = (দশমিক *) malloc(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) malloc(x_sz_f16);

    জন্য (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (দশমিক)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_to_fp16(ctx, quant);

    ggml_vk_quantize_তথ্য(x, qx, ne, quant);
    ggml_vk_dequantize_তথ্য(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_বাফার_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ধ্রুবক std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, সত্য, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    দ্বিগুণ ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_বাফার_read(x_buf, 0, x_chk, x_sz_f16);

    পূর্ণসংখ্যা first_err = -1;

    দ্বিগুণ avg_err = 0.0;
    জন্য (size_t i = 0; i < ne; i++) {
        দ্বিগুণ error = std::fabs(x_ref[i] - ggml_fp16_to_fp32(x_chk[i]));
        avg_err += error;

        যদি (first_err < 0 && error > 0.05) {
            first_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::endl;

    যদি (avg_err > 0.1) {
        std::cerr << "first_error = " << first_err << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        জন্য (পূর্ণসংখ্যা i = std::max(0, first_err - 5); i < std::min((পূর্ণসংখ্যা)ne, first_err + 5); i++) {
            std::cerr << ggml_fp16_to_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
        জন্য (পূর্ণসংখ্যা i = std::max(0, first_err - 5); i < std::min((পূর্ণসংখ্যা)ne, first_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::endl;
    }

    ggml_vk_destroy_বাফার(x_buf);
    ggml_vk_destroy_বাফার(qx_buf);

    মুক্ত(x);
    মুক্ত(qx);
    মুক্ত(x_ref);
    মুক্ত(x_chk);
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

শূন্য ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t অঙ্ক_it, size_t split_k, size_t shader_আকার, ggml_type quant, যৌক্তিক mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << অঙ্ক_it << ", " << split_k << ", " << ggml_type_name(quant) << ")");
    ধ্রুবক size_t x_ne = m * k * batch;
    ধ্রুবক size_t y_ne = k * n * batch;
    ধ্রুবক size_t d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    যৌক্তিক f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_চাবি dq_চাবি{quant, b_type, মিথ্যা, f16acc};
    auto dq_it = ctx->device->pipeline_matmul.find(dq_চাবি);
    যদি (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        যদি (f16acc) {
            dq_চাবি.f16acc = মিথ্যা;
            dq_it = ctx->device->pipeline_matmul.find(dq_চাবি);
        }
    }
    যদি (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::endl;
        ফেরত;
    }
    auto& dq_configs = dq_it->second;
    যদি (shader_আকার >= (পূর্ণসংখ্যা)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_আকার << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::endl;
        ফেরত;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_আকার);
    vk_pipeline p = dq_configs[shader_আকার].aligned ? dq_configs[shader_আকার].aligned : dq_configs[shader_আকার].unaligned;

    ধ্রুবক size_t kpad = mmq ? 0 : ggml_vk_align_আকার(k, dq_configs[shader_আকার].align);

    যদি (mmq || k != kpad) {
        p = dq_configs[shader_আকার].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_আকার);
    }

    যদি (p == nullptr) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::endl;
        ফেরত;
    }

    ধ্রুবক size_t x_sz = আকার(দশমিক) * x_ne;
    ধ্রুবক size_t y_sz = আকার(দশমিক) * y_ne;
    ধ্রুবক size_t qx_sz = x_ne * ggml_type_আকার(quant)/ggml_blck_আকার(quant);
    ধ্রুবক size_t qy_sz = mmq ? y_ne * ggml_type_আকার(GGML_TYPE_Q8_1)/ggml_blck_আকার(GGML_TYPE_Q8_1) : y_sz;
    ধ্রুবক size_t d_sz = আকার(দশমিক) * d_ne;
    দশমিক * x = (দশমিক *) malloc(x_sz);
    দশমিক * y = (দশমিক *) malloc(y_sz);
    শূন্য * qx = malloc(qx_sz);
    vk_বাফার qx_buf = ggml_vk_create_বাফার_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_বাফার y_buf = ggml_vk_create_বাফার_check(ctx->device, y_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_বাফার qy_buf = ggml_vk_create_বাফার_check(ctx->device, qy_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_বাফার d_buf = ggml_vk_create_বাফার_check(ctx->device, d_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    দশমিক * d = (দশমিক *) malloc(d_sz);
    দশমিক * d_chk = (দশমিক *) malloc(d_sz);

    জন্য (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (দশমিক)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_তথ্য(x, qx, x_ne, quant);

    জন্য (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (দশমিক)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    যদি (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, অঙ্ক_it);

        যদি (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < আকার(দশমিক) * d_ne * split_k) {
            // Resize buffer
            যদি (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_বাফার(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_বাফার_check(ctx->device, আকার(দশমিক) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }
    যদি (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, অঙ্ক_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_বাফার_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_বাফার_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    যদি (mmq) {
        জন্য (size_t i = 0; i < অঙ্ক_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_আকার_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_আকার(GGML_TYPE_Q8_1)) * ggml_type_আকার(GGML_TYPE_Q8_1), m * আকার(দশমিক)
            );
        }
    } অন্যথা {
        জন্য (size_t i = 0; i < অঙ্ক_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_আকার_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                k * আকার(দশমিক), m * আকার(দশমিক)
            );
        }
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, সত্য, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    দ্বিগুণ time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_বাফার_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->তথ্য = qx;
    src1_ggml->তথ্য = y;
    tensor_ggml->তথ্য = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    দ্বিগুণ avg_err = 0.0;
    পূর্ণসংখ্যা first_err_n = -1;
    পূর্ণসংখ্যা first_err_m = -1;
    পূর্ণসংখ্যা first_err_b = -1;

    জন্য (size_t i = 0; i < m*n*batch; i++) {
        দ্বিগুণ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        যদি ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    দ্বিগুণ tflops = 2.0*m*n*k*batch*অঙ্ক_it / (time_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    যদি (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time_ms / অঙ্ক_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    যদি (avg_err > 0.01 || std::isnan(avg_err)) {
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

        যদি (split_k > 1) {
            দশমিক * split_k_buf = (দশমিক *) malloc(আকার(দশমিক) * d_ne * split_k);
            ggml_vk_বাফার_read(ctx->prealloc_split_k, 0, split_k_buf, আকার(দশমিক) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            মুক্ত(split_k_buf);
        }
    }

    ggml_vk_destroy_বাফার(qx_buf);
    ggml_vk_destroy_বাফার(y_buf);
    ggml_vk_destroy_বাফার(qy_buf);
    ggml_vk_destroy_বাফার(d_buf);

    মুক্ত(x);
    মুক্ত(qx);
    মুক্ত(y);
    মুক্ত(d);
    মুক্ত(d_chk);
}
#endif

int64_t ggml_vk_get_op_batch_আকার(ধ্রুবক ggml_tensor * op) {
    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_GET_ROWS:
            ফেরত 0;
        ক্ষেত্র GGML_OP_MUL_MAT:
            ফেরত op->ne[1];
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
        ক্ষেত্র GGML_OP_ROPE:
        ক্ষেত্র GGML_OP_ROPE_BACK:
            ফেরত op->ne[2];
        পূর্বনির্ধারিত:
            ফেরত ggml_nrows(op);
    }
}

#ifdef GGML_VULKAN_CHECK_RESULTS
স্থির শূন্য ggml_vk_print_graph_origin(ধ্রুবক ggml_tensor * tensor, std::vector<ধ্রুবক ggml_tensor *>& done, পূর্ণসংখ্যা level = 0) {
    যদি (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) {
        ফেরত;
    }
    জন্য (পূর্ণসংখ্যা j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;

    done.push_back(tensor);

    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        যদি (tensor->src[i] != nullptr) {
            ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

স্থির শূন্য ggml_vk_print_tensor_area(ধ্রুবক ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, পূর্ণসংখ্যা i0, পূর্ণসংখ্যা i1, পূর্ণসংখ্যা i2, পূর্ণসংখ্যা i3) {
    যদি (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        ফেরত;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    জন্য (পূর্ণসংখ্যা idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    জন্য (পূর্ণসংখ্যা idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        জন্য (পূর্ণসংখ্যা idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            যদি (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                দশমিক val;
                যদি (tensor->type == GGML_TYPE_F32) {
                    val = *(ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } অন্যথা যদি (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ধ্রুবক ggml_fp16_t *) ((ধ্রুবক অক্ষর *) তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } অন্যথা যদি (tensor->type == GGML_TYPE_I32) {
                    val = *(ধ্রুবক int32_t *) ((ধ্রুবক অক্ষর *) তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } অন্যথা {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } অন্যথা {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

স্থির শূন্য ggml_vk_print_tensor(ধ্রুবক ggml_tensor * tensor, ধ্রুবক অক্ষর * name) {
    শূন্য * tensor_তথ্য = tensor->তথ্য;

    ধ্রুবক যৌক্তিক is_gpu = tensor->বাফার != nullptr && ggml_backend_বাফার_is_vk(tensor->বাফার);

    যদি (is_gpu) {
        ধ্রুবক size_t tensor_আকার = ggml_nbytes(tensor);
        tensor_তথ্য = malloc(tensor_আকার);

        ggml_backend_vk_বাফার_context * buf_ctx = (ggml_backend_vk_বাফার_context *)tensor->বাফার->context;

        vk_বাফার বাফার_gpu = buf_ctx->dev_বাফার;
        ggml_vk_বাফার_read(বাফার_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_তথ্য, tensor_আকার);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::endl;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
    যদি (tensor->src[0] != nullptr) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::endl;
    }
    যদি (tensor->src[1] != nullptr) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::endl;
    }
    std::cerr << std::endl << "Result:" << std::endl;
    ggml_vk_print_tensor_area(tensor, tensor_তথ্য, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<ধ্রুবক ggml_tensor *> done;
    ggml_vk_print_graph_origin(tensor, done);

    যদি (is_gpu) {
        মুক্ত(tensor_তথ্য);
    }
}

শূন্য * comp_ফলাফল;
size_t comp_আকার;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_গণক = 0;
শূন্য ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, পূর্ণসংখ্যা tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->অঙ্ক_additional_fused_ops];
    যদি (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        ফেরত;
    }

    check_গণক++;
    যদি (!(vk_output_tensor > 0 && vk_output_tensor == check_গণক) && check_গণক <= vk_skip_checks) {
        ফেরত;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    কাঠামো ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ মিথ্যা,
    };

    কাঠামো ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<কাঠামো ggml_tensor *, GGML_MAX_SRC> src_clone = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    ধ্রুবক অক্ষর * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<শূন্য *> cloned_mallocs;

    কাঠামো ggml_tensor * tensor_clone = nullptr;

    জন্য (পূর্ণসংখ্যা f = 0; f < ctx->অঙ্ক_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            যদি (srci == nullptr) {
                চালু;
            }
            // If a src tensor has been cloned, use that one
            auto it = cloned_tensors.find(srci);
            যদি (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                চালু;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            size_t srci_আকার = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            শূন্য *src_বাফার = malloc(srci_আকার);
            cloned_mallocs.push_back(src_বাফার);

            srci_clone->তথ্য = src_বাফার;
            যদি (ggml_backend_বাফার_is_host(srci->বাফার)) {
                memcpy(srci_clone->তথ্য, srci->তথ্য, srci_আকার);
                memcpy(srci_clone->nb, srci->nb, আকার(size_t) * GGML_MAX_DIMS);
            } অন্যথা যদি (ggml_backend_বাফার_is_vk(srci->বাফার)) {
                ggml_backend_vk_বাফার_context * buf_ctx = (ggml_backend_vk_বাফার_context *)srci->বাফার->context;
                vk_বাফার& বাফার_gpu = buf_ctx->dev_বাফার;
                uint64_t offset = vk_tensor_offset(srci) + srci->view_offs;
                যদি (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    জন্য (পূর্ণসংখ্যা i3 = 0; i3 < srci->ne[3]; i3++) {
                        জন্য (পূর্ণসংখ্যা i2 = 0; i2 < srci->ne[2]; i2++) {
                            ধ্রুবক পূর্ণসংখ্যা idx = i3*srci->ne[2] + i2;
                            ggml_vk_বাফার_read(বাফার_gpu, offset + idx * srci->nb[2], ((অক্ষর *)srci_clone->তথ্য + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    জন্য (পূর্ণসংখ্যা i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } অন্যথা {
                    যদি (offset + srci_আকার >= বাফার_gpu->size) {
                        srci_আকার = বাফার_gpu->size - offset;
                    }
                    ggml_vk_বাফার_read(বাফার_gpu, offset, srci_clone->তথ্য, srci_আকার);
                    memcpy(srci_clone->nb, srci->nb, আকার(size_t) * GGML_MAX_DIMS);
                }
            } অন্যথা {
                GGML_ABORT("fatal error");
            }

            যদি (vk_output_tensor > 0 && vk_output_tensor == check_গণক) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        যদি (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ধ্রুবক দশমিক * params = (ধ্রুবক দশমিক *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            যদি (src_clone[4]) {
                ggml_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]);
            }
        } অন্যথা যদি (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } অন্যথা যদি (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(পূর্ণসংখ্যা *)tensor->op_params);
        } অন্যথা যদি (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_mode) tensor->op_params[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_SCALE) {
            ধ্রুবক দশমিক * params = (ধ্রুবক দশমিক *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_ARANGE) {
            ধ্রুবক দশমিক start = ggml_get_op_params_f32(tensor, 0);
            ধ্রুবক দশমিক stop = ggml_get_op_params_f32(tensor, 1);
            ধ্রুবক দশমিক step = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, step);
        } অন্যথা যদি (tensor->op == GGML_OP_FILL) {
            ধ্রুবক দশমিক value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } অন্যথা যদি (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } অন্যথা যদি (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_CLAMP) {
            ধ্রুবক দশমিক * params = (ধ্রুবক দশমিক *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } অন্যথা যদি (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } অন্যথা যদি (tensor->op == GGML_OP_REPEAT_BACK) {
            tensor_clone = ggml_repeat_back(ggml_ctx, src_clone[0], tensor);
        } অন্যথা যদি (tensor->op == GGML_OP_ADD) {
            tensor_clone = ggml_add(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } অন্যথা যদি (tensor->op == GGML_OP_SET) {
            tensor_clone = ggml_set(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } অন্যথা যদি (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(দশমিক *)tensor->op_params);
        } অন্যথা যদি (tensor->op == GGML_OP_GROUP_NORM) {
            ধ্রুবক দশমিক * float_params = (ধ্রুবক দশমিক *)tensor->op_params;
            tensor_clone = ggml_group_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(দশমিক *)tensor->op_params);
        } অন্যথা যদি (tensor->op == GGML_OP_RMS_NORM_BACK) {
            ধ্রুবক দশমিক eps = ((দশমিক *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_back(ggml_ctx, src_clone[0], src_clone[1], eps);
        } অন্যথা যদি (tensor->op == GGML_OP_SILU_BACK) {
            tensor_clone = ggml_silu_back(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_L2_NORM) {
            ধ্রুবক দশমিক eps = ((দশমিক *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } অন্যথা যদি (tensor->op == GGML_OP_SOFT_MAX) {
            যদি (tensor->src[1] != nullptr) {
                ধ্রুবক দশমিক * params = (ধ্রুবক দশমিক *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } অন্যথা {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } অন্যথা যদি (tensor->op == GGML_OP_SOFT_MAX_BACK) {
            tensor_clone = ggml_soft_max_ext_back(ggml_ctx, src_clone[0], src_clone[1], ((দশমিক *)tensor->op_params)[0], ((দশমিক *)tensor->op_params)[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_DIAG_MASK_INF) {
            tensor_clone = ggml_diag_mask_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_BACK) {
            ধ্রুবক পূর্ণসংখ্যা n_dims      = ((int32_t *) tensor->op_params)[1];
            ধ্রুবক পূর্ণসংখ্যা mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            ধ্রুবক পূর্ণসংখ্যা n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            ধ্রুবক দশমিক freq_base       = ((দশমিক *) tensor->op_params)[5];
            ধ্রুবক দশমিক freq_scale      = ((দশমিক *) tensor->op_params)[6];
            ধ্রুবক দশমিক ext_factor      = ((দশমিক *) tensor->op_params)[7];
            ধ্রুবক দশমিক attn_factor     = ((দশমিক *) tensor->op_params)[8];
            ধ্রুবক দশমিক beta_fast       = ((দশমিক *) tensor->op_params)[9];
            ধ্রুবক দশমিক beta_slow       = ((দশমিক *) tensor->op_params)[10];
            যদি (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                যদি (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } অন্যথা {
                    tensor_clone = ggml_rope_multi_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            } অন্যথা {
                যদি (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } অন্যথা {
                    tensor_clone = ggml_rope_ext_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            }
            ধ্রুবক পূর্ণসংখ্যা n_offs = ((int32_t *) tensor->op_params)[15];
            যদি (n_offs != 0) {
                tensor_clone = ggml_rope_set_offset(tensor_clone, n_offs);
            }
        } অন্যথা যদি (tensor->op == GGML_OP_UNARY) {
            নির্বাচন (ggml_get_unary_op(tensor)) {
            ক্ষেত্র GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_GELU_QUICK:
                tensor_clone = ggml_gelu_quick(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_set_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_set_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_set_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_set_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_STEP:
                tensor_clone = ggml_step(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_ROUND:
                tensor_clone = ggml_round(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                বিরতি;
            ক্ষেত্র GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                বিরতি;
            পূর্বনির্ধারিত:
                std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
                GGML_ABORT("fatal error");
            }
        } অন্যথা যদি (tensor->op == GGML_OP_GLU) {
            যদি (src_clone[1] == nullptr) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } অন্যথা {
                tensor_clone = ggml_glu_split(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_set_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_set_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } অন্যথা যদি (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            যদি (tensor->src[1] == nullptr) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } অন্যথা {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } অন্যথা যদি (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } অন্যথা যদি (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } অন্যথা যদি (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_PERMUTE) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_permute(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } অন্যথা যদি (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_GET_ROWS) {
            tensor_clone = ggml_get_rows(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_sort_order) *(পূর্ণসংখ্যা *)tensor->op_params);
        } অন্যথা যদি (tensor->op == GGML_OP_শীর্ষ_K) {
            tensor_clone = ggml_শীর্ষ_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_যোগ) {
            tensor_clone = ggml_যোগ(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_যোগ_ROWS) {
            tensor_clone = ggml_যোগ_rows(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } অন্যথা যদি (tensor->op == GGML_OP_DSV4_HC_PRE) {
            যদি (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } অন্যথা {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } অন্যথা যদি (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } অন্যথা যদি (tensor->op == GGML_OP_MEAN) {
            tensor_clone = ggml_mean(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_BACK) {
            tensor_clone = ggml_cross_entropy_loss_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } অন্যথা যদি (tensor->op == GGML_OP_গুনতি_EQUAL) {
            tensor_clone = ggml_গুনতি_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], সত্য, সত্য, মিথ্যা);
        } অন্যথা যদি (tensor->op == GGML_OP_IM2COL) {
            ধ্রুবক int32_t s0 = tensor->op_params[0];
            ধ্রুবক int32_t s1 = tensor->op_params[1];
            ধ্রুবক int32_t p0 = tensor->op_params[2];
            ধ্রুবক int32_t p1 = tensor->op_params[3];
            ধ্রুবক int32_t d0 = tensor->op_params[4];
            ধ্রুবক int32_t d1 = tensor->op_params[5];

            ধ্রুবক যৌক্তিক is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } অন্যথা যদি (tensor->op == GGML_OP_IM2COL_3D) {
            ধ্রুবক int32_t s0 = tensor->op_params[0];
            ধ্রুবক int32_t s1 = tensor->op_params[1];
            ধ্রুবক int32_t s2 = tensor->op_params[2];
            ধ্রুবক int32_t p0 = tensor->op_params[3];
            ধ্রুবক int32_t p1 = tensor->op_params[4];
            ধ্রুবক int32_t p2 = tensor->op_params[5];
            ধ্রুবক int32_t d0 = tensor->op_params[6];
            ধ্রুবক int32_t d1 = tensor->op_params[7];
            ধ্রুবক int32_t d2 = tensor->op_params[8];
            ধ্রুবক int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } অন্যথা যদি (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            ধ্রুবক int32_t dim = tensor->op_params[0];
            ধ্রুবক int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } অন্যথা যদি (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            ধ্রুবক int32_t s0 = tensor->op_params[0];
            ধ্রুবক int32_t p0 = tensor->op_params[1];
            ধ্রুবক int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } অন্যথা যদি (tensor->op == GGML_OP_COL2IM_1D) {
            ধ্রুবক int32_t stride = tensor->op_params[0];
            ধ্রুবক int32_t oc     = tensor->op_params[1];
            ধ্রুবক int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } অন্যথা যদি (tensor->op == GGML_OP_POOL_1D) {
            গণনা ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ধ্রুবক int32_t k0 = tensor->op_params[1];
            ধ্রুবক int32_t s0 = tensor->op_params[2];
            ধ্রুবক int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } অন্যথা যদি (tensor->op == GGML_OP_POOL_2D) {
            গণনা ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ধ্রুবক int32_t k0 = tensor->op_params[1];
            ধ্রুবক int32_t k1 = tensor->op_params[2];
            ধ্রুবক int32_t s0 = tensor->op_params[3];
            ধ্রুবক int32_t s1 = tensor->op_params[4];
            ধ্রুবক int32_t p0 = tensor->op_params[5];
            ধ্রুবক int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } অন্যথা যদি (tensor->op == GGML_OP_CONV_2D) {
            ধ্রুবক int32_t s0 = tensor->op_params[0];
            ধ্রুবক int32_t s1 = tensor->op_params[1];
            ধ্রুবক int32_t p0 = tensor->op_params[2];
            ধ্রুবক int32_t p1 = tensor->op_params[3];
            ধ্রুবক int32_t d0 = tensor->op_params[4];
            ধ্রুবক int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } অন্যথা যদি (tensor->op == GGML_OP_CONV_3D) {
            ধ্রুবক int32_t s0 = tensor->op_params[0];
            ধ্রুবক int32_t s1 = tensor->op_params[1];
            ধ্রুবক int32_t s2 = tensor->op_params[2];
            ধ্রুবক int32_t p0 = tensor->op_params[3];
            ধ্রুবক int32_t p1 = tensor->op_params[4];
            ধ্রুবক int32_t p2 = tensor->op_params[5];
            ধ্রুবক int32_t d0 = tensor->op_params[6];
            ধ্রুবক int32_t d1 = tensor->op_params[7];
            ধ্রুবক int32_t d2 = tensor->op_params[8];
            ধ্রুবক int32_t IC = tensor->op_params[9];
            ধ্রুবক int32_t N  = tensor->op_params[10];
            ধ্রুবক int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } অন্যথা যদি (tensor->op == GGML_OP_CONV_2D_DW) {
            ধ্রুবক int32_t s0 = tensor->op_params[0];
            ধ্রুবক int32_t s1 = tensor->op_params[1];
            ধ্রুবক int32_t p0 = tensor->op_params[2];
            ধ্রুবক int32_t p1 = tensor->op_params[3];
            ধ্রুবক int32_t d0 = tensor->op_params[4];
            ধ্রুবক int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } অন্যথা যদি (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            ধ্রুবক int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } অন্যথা যদি (tensor->op == GGML_OP_LEAKY_RELU) {
            ধ্রুবক দশমিক * op_params = (ধ্রুবক দশমিক *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], মিথ্যা);
        } অন্যথা যদি (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } অন্যথা যদি (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } অন্যথা যদি (tensor->op == GGML_OP_GATED_LINEAR_ATTN) {
            ধ্রুবক দশমিক * op_params = (ধ্রুবক দশমিক *)tensor->op_params;
            tensor_clone = ggml_gated_linear_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } অন্যথা যদি (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } অন্যথা যদি (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } অন্যথা যদি (tensor->op == GGML_OP_OPT_STEP_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } অন্যথা যদি (tensor->op == GGML_OP_OPT_STEP_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } অন্যথা যদি (tensor->op == GGML_OP_ADD_ID) {
            tensor_clone = ggml_add_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } অন্যথা যদি (tensor->op == GGML_OP_SSM_SCAN) {
            ধ্রুবক int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_scan(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } অন্যথা যদি (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } অন্যথা যদি (tensor->op == GGML_OP_ROLL) {
            ধ্রুবক int32_t s0 = tensor->op_params[0];
            ধ্রুবক int32_t s1 = tensor->op_params[1];
            ধ্রুবক int32_t s2 = tensor->op_params[2];
            ধ্রুবক int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        অন্যথা {
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    যদি (vk_output_tensor > 0 && vk_output_tensor == check_গণক) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_আকার = ggml_nbytes(tensor_clone);

    comp_ফলাফল = malloc(comp_আকার);
    memcpy(comp_ফলাফল, tensor_clone->তথ্য, comp_আকার);
    memcpy(comp_nb, tensor_clone->nb, আকার(size_t) * GGML_MAX_DIMS);

    জন্য (auto m : cloned_mallocs) {
        মুক্ত(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

শূন্য ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, পূর্ণসংখ্যা tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->অঙ্ক_additional_fused_ops];
    যদি (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        ফেরত;
    }

    যদি (!(vk_output_tensor > 0 && vk_output_tensor == check_গণক) && check_গণক <= vk_skip_checks) {
        ফেরত;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    শূন্য * tensor_তথ্য = tensor->তথ্য;

    যদি (ggml_backend_বাফার_is_vk(tensor->বাফার)) {
        size_t tensor_আকার = ggml_nbytes(tensor);
        tensor_তথ্য = malloc(tensor_আকার);

        ggml_backend_vk_বাফার_context * buf_ctx = (ggml_backend_vk_বাফার_context *)tensor->বাফার->context;

        vk_বাফার& বাফার_gpu = buf_ctx->dev_বাফার;
        uint64_t offset = vk_tensor_offset(tensor) + tensor->view_offs;
        যদি (offset + tensor_আকার >= বাফার_gpu->size) {
            tensor_আকার = বাফার_gpu->size - offset;
        }

        ggml_vk_বাফার_read(বাফার_gpu, offset, tensor_তথ্য, tensor_আকার);
    }

    দশমিক first_error_ফলাফল = -1.0f;
    দশমিক first_error_correct = -1.0f;
    std::array<পূর্ণসংখ্যা, 4> first_error = { -1, -1, -1, -1 };
    দ্বিগুণ avg_err = 0.0;
    size_t গণক = 0;

    জন্য (পূর্ণসংখ্যা i3 = 0; i3 < tensor->ne[3]; i3++) {
        জন্য (পূর্ণসংখ্যা i2 = 0; i2 < tensor->ne[2]; i2++) {
            জন্য (পূর্ণসংখ্যা i1 = 0; i1 < tensor->ne[1]; i1++) {
                জন্য (পূর্ণসংখ্যা i0 = 0; i0 < tensor->ne[0]; i0++) {
                    ধ্রুবক যৌক্তিক বাফার_আকার_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_আকার;
                    দশমিক correct = 0.0f;
                    দশমিক ফলাফল = 0.0f;

                    যদি (বাফার_আকার_fit) {
                        যদি (tensor->type == GGML_TYPE_F32) {
                            correct = *(দশমিক *) ((অক্ষর *) comp_ফলাফল + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ফলাফল  = *(দশমিক *) ((অক্ষর *) tensor_তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } অন্যথা যদি (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((অক্ষর *) comp_ফলাফল + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ফলাফল  = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((অক্ষর *) tensor_তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } অন্যথা যদি (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((অক্ষর *) comp_ফলাফল + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ফলাফল  = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((অক্ষর *) tensor_তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } অন্যথা যদি (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((অক্ষর *) comp_ফলাফল + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ফলাফল  = *(int32_t *) ((অক্ষর *) tensor_তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } অন্যথা যদি (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((অক্ষর *) comp_ফলাফল + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ফলাফল  = *(int64_t *) ((অক্ষর *) tensor_তথ্য + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } অন্যথা {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::endl;
                        }
                    } অন্যথা {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::endl;
                        GGML_ABORT("fatal error");
                    }

                    যদি ((std::isnan(correct) != std::isnan(ফলাফল)) || (std::isinf(correct) != std::isinf(ফলাফল)) || !বাফার_আকার_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << ফলাফল << " correct=" << correct << " avg_err=" << (avg_err / গণক) << std::endl;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
                        যদি (src0 != nullptr) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
                        }
                        যদি (src1 != nullptr) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
                        }
                        যদি (src2 != nullptr) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
                        }
                        যদি (src3 != nullptr) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
                        }
                        std::cerr << "First error: result=" << first_error_ফলাফল << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
                        std::cerr << std::endl << "Result:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, tensor_তথ্য, i0, i1, i2, i3);
                        std::cerr << std::endl << "Correct:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, comp_ফলাফল, i0, i1, i2, i3);
                        std::cerr << std::endl;
                        std::vector<ধ্রুবক ggml_tensor *> done;
                        ggml_vk_print_graph_origin(tensor, done);
                        GGML_ABORT("fatal error");
                    }
                    ধ্রুবক দ্বিগুণ denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    যদি (first_error[0] == -1 && std::fabs(correct - ফলাফল) / denom > 0.5) {
                        first_error[0] = i0;
                        first_error[1] = i1;
                        first_error[2] = i2;
                        first_error[3] = i3;
                        first_error_ফলাফল = ফলাফল;
                        first_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    যদি (!std::isinf(correct) && !std::isinf(ফলাফল) && !std::isnan(correct) && !std::isnan(ফলাফল)) {
                        avg_err += std::fabs(correct - ফলাফল) / denom;
                    }
                    গণক++;
                }
            }
        }
    }

    avg_err /= গণক;

    যদি (vk_output_tensor > 0 && vk_output_tensor == check_গণক) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_গণক << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        যদি (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        যদি (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        যদি (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        যদি (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_ফলাফল << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_তথ্য, 5, 5, 0, 0);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_ফলাফল, 5, 5, 0, 0);
        std::cerr << std::endl;
        std::vector<ধ্রুবক ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
    }

    যদি (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_গণক << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        যদি (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        যদি (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        যদি (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        যদি (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_ফলাফল << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_তথ্য, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_ফলাফল, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl;
        std::vector<ধ্রুবক ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
        GGML_ABORT("fatal error");
    } অন্যথা {
        std::cerr << check_গণক << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::endl;
    }

    মুক্ত(comp_ফলাফল);
    comp_ফলাফল = nullptr;
    comp_আকার = 0;

    যদি (ggml_backend_বাফার_is_vk(tensor->বাফার)) {
        মুক্ত(tensor_তথ্য);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#endif

