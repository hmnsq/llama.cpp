#అన్తర్భావ "ggml-vulkan-common.h"

తార్కిక vk_జ్ఞాపకం_logger_enabled = అబద్ధం;

తార్కిక vk_perf_logger_enabled = అబద్ధం;

తార్కిక vk_perf_logger_concurrent = అబద్ధం;

తార్కిక vk_enable_sync_logger = అబద్ధం;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

శూన్య vk_జ్ఞాపకం_logger::log_కేటాయింపు(vk_బఫరు_ref buf_ref, size_t size) {
    ఒకవేళ (!vk_జ్ఞాపకం_logger_enabled) {
        తిరిగి;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_బఫరు buf = buf_ref.తాళం();
    స్థిరరాశి తార్కిక device = తార్కిక(buf->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal);
    స్థిరరాశి std::string type = device ? "device" : "host";
    allocations[buf->బఫరు] = size;
    మొత్తం_device += device ? size : 0;
    మొత్తం_host += device ? 0 : size;
    VK_LOG_జ్ఞాపకం(buf->device->name << ": +" << format_కొలత(size) << " " << type << " at " << buf->బఫరు << ". Total device: " << format_కొలత(మొత్తం_device) << ", total host: " << format_కొలత(మొత్తం_host));
}

శూన్య vk_జ్ఞాపకం_logger::log_deallocation(vk_బఫరు_ref buf_ref) {
    ఒకవేళ (buf_ref.expired() || buf_ref.తాళం()->size == 0 || !vk_జ్ఞాపకం_logger_enabled) {
        తిరిగి;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_బఫరు buf = buf_ref.తాళం();
    స్థిరరాశి తార్కిక device = తార్కిక(buf->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal);
    std::string type = device ? "device" : "host";
    స్వయంచాలక it = allocations.find(buf->బఫరు);
    ఒకవేళ (it != allocations.end()) {
        మొత్తం_device -= device ? it->second : 0;
        మొత్తం_host -= device ? 0 : it->second;
        VK_LOG_జ్ఞాపకం(buf->device->name << ": -" << format_కొలత(it->second) << " " << type << " at " << buf->బఫరు << ". Total device: " << format_కొలత(మొత్తం_device) << ", total host: " << format_కొలత(మొత్తం_host));
        allocations.erase(it);
    } లేకపోతే {
        VK_LOG_జ్ఞాపకం("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->బఫరు);
    }
}

#నిర్వచితమైతే GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

స్థిర శూన్య ggml_vk_print_tensor(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి అక్షరం * name);
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_VULKAN_నిరంతరం_TESTS
స్థిర శూన్య ggml_vk_print_మాతృక_area(స్థిరరాశి శూన్య * దత్తాంశం, ggml_type type, పూర్ణాంకం ne0, పూర్ణాంకం ne1, పూర్ణాంకం i0, పూర్ణాంకం i1, పూర్ణాంకం i2) {
    ఒకవేళ (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        తిరిగి;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    కొరకు (పూర్ణాంకం idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    కొరకు (పూర్ణాంకం idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        కొరకు (పూర్ణాంకం idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ఒకవేళ (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                దశాంశం val;
                ఒకవేళ (type == GGML_TYPE_F32) {
                    val = *((స్థిరరాశి దశాంశం *) దత్తాంశం + i2*ne1*ne0 + idx1*ne0 + idx0);
                } లేకపోతే ఒకవేళ (type == GGML_TYPE_F16) {
                    val = ggml_fp16_గమ్యం_fp32(*((స్థిరరాశి ggml_fp16_t *) దత్తాంశం + i2*ne1*ne0 + idx1*ne0 + idx0));
                } లేకపోతే {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } లేకపోతే {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

మూస <రకం_పేరు X_TYPE, రకం_పేరు Y_TYPE>
శూన్య ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t అంకె_it, పూర్ణాంకం విభజించు_k, పూర్ణాంకం shader_కొలత) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << అంకె_it << ", " << విభజించు_k << ", " << shader_కొలత << ")");
    స్థిరరాశి size_t x_ne = m * k * batch;
    స్థిరరాశి size_t y_ne = k * n * batch;
    స్థిరరాశి size_t d_ne = m * n * batch;

    ggml_type x_type = std::is_same<దశాంశం, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<దశాంశం, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_కీలకం mm_test_కీలకం{x_type, y_type, అబద్ధం, అబద్ధం};
    స్వయంచాలక mm_test_it = ctx->device->pipeline_matmul.find(mm_test_కీలకం);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.ఖాళీలు());
    స్వయంచాలక& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_కొలత >= 0 && shader_కొలత < (పూర్ణాంకం)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_కొలత);
    vk_pipeline p = mm_test_configs[shader_కొలత].aligned ? mm_test_configs[shader_కొలత].aligned : mm_test_configs[shader_కొలత].unaligned;

    స్థిరరాశి size_t kpad = ggml_vk_align_కొలత(k, mm_test_configs[shader_కొలత].align);

    ఒకవేళ (k != kpad) {
        p = mm_test_configs[shader_కొలత].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_కొలత);
    }

    ఒకవేళ (విభజించు_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_విభజించు_k_reduce, అంకె_it);

        ఒకవేళ (ctx->prealloc_విభజించు_k == శూన్యసూచిక || ctx->prealloc_విభజించు_k->size < పరిమాణం(దశాంశం) * d_ne * విభజించు_k) {
            // Resize buffer
            ఒకవేళ (ctx->prealloc_విభజించు_k != శూన్యసూచిక) {
                ggml_vk_destroy_బఫరు(ctx->prealloc_విభజించు_k);
            }
            ctx->prealloc_విభజించు_k = ggml_vk_create_బఫరు_check(ctx->device, పరిమాణం(దశాంశం) * d_ne * విభజించు_k, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_బఫరు d_X = ggml_vk_create_బఫరు_check(ctx->device, పరిమాణం(X_TYPE) * x_ne, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    vk_బఫరు d_Y = ggml_vk_create_బఫరు_check(ctx->device, పరిమాణం(Y_TYPE) * y_ne, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    vk_బఫరు d_D = ggml_vk_create_బఫరు_check(ctx->device, పరిమాణం(దశాంశం) * d_ne, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(Y_TYPE) * y_ne);
    దశాంశం* d = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(దశాంశం) * d_ne);

    కొరకు (size_t i = 0; i < x_ne; i++) {
        ఒకవేళ (std::is_same<దశాంశం, X_TYPE>()) {
            x[i] = (rand() / (దశాంశం)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } లేకపోతే ఒకవేళ (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_గమ్యం_fp16((rand() / (దశాంశం)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } లేకపోతే {
            GGML_ABORT("fatal error");
        }
    }
    కొరకు (size_t i = 0; i < y_ne; i++) {
        ఒకవేళ (std::is_same<దశాంశం, Y_TYPE>()) {
            y[i] = (rand() / (దశాంశం)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } లేకపోతే ఒకవేళ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_గమ్యం_fp16((rand() / (దశాంశం)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } లేకపోతే {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_బఫరు_write(d_X, 0, x, పరిమాణం(X_TYPE) * k * m * batch);
    ggml_vk_బఫరు_write(d_Y, 0, y, పరిమాణం(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    కొరకు (size_t i = 0; i < అంకె_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_విభజించు_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            విభజించు_k, batch, batch, batch, 1, 1, n,
            k * పరిమాణం(Y_TYPE), m * పరిమాణం(దశాంశం)
        );
    }
    ggml_vk_ctx_చివర(subctx);

    స్వయంచాలక begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, నిజం, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_వరుస_command_pools_cleanup(ctx->device);

    స్వయంచాలక end = std::chrono::high_resolution_clock::now();
    ద్విగుణ సమయం = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_బఫరు_read(d_D, 0, d, పరిమాణం(దశాంశం) * d_ne);

    దశాంశం * d_chk = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(దశాంశం) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    ఒకవేళ (std::is_same<దశాంశం, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } లేకపోతే ఒకవేళ (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }
    ఒకవేళ (std::is_same<దశాంశం, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } లేకపోతే ఒకవేళ (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->దత్తాంశం = x;
    src1_ggml->దత్తాంశం = y;
    tensor_ggml->దత్తాంశం = d_chk;

    ggml_cgraph * cgraph = ggml_new_లేఖాచిత్రం(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_లేఖాచిత్రం_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ద్విగుణ avg_err = 0.0;
    పూర్ణాంకం మొదటి_err_n = -1;
    పూర్ణాంకం మొదటి_err_m = -1;
    పూర్ణాంకం మొదటి_err_b = -1;

    కొరకు (size_t i = 0; i < m*n*batch; i++) {
        ద్విగుణ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ఒకవేళ ((err > 0.05f || std::isnan(err)) && మొదటి_err_n == -1) {
            మొదటి_err_b = i / (m * n);
            మొదటి_err_n = (i % (m * n)) / m;
            మొదటి_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ద్విగుణ tflops = 2.0*m*n*k*batch*అంకె_it / (సమయం / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << విభజించు_k << " matmul " << సమయం / అంకె_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::పంక్తి_ముగింపు;

    ఒకవేళ (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << మొదటి_err_m << " n = " << మొదటి_err_n << " b = " << మొదటి_err_b << std::పంక్తి_ముగింపు;
        std::cerr << "Actual result: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        ggml_vk_print_మాతృక_area(d, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);
        std::cerr << "Expected result: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        ggml_vk_print_మాతృక_area(d_chk, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

        ఒకవేళ (విభజించు_k > 1) {
            దశాంశం * విభజించు_k_buf = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(దశాంశం) * d_ne * విభజించు_k);
            ggml_vk_బఫరు_read(ctx->prealloc_విభజించు_k, 0, విభజించు_k_buf, పరిమాణం(దశాంశం) * d_ne * విభజించు_k);

            std::cerr << "d_buf0: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            std::cerr << "d_buf1: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf + d_ne, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            std::cerr << "d_buf2: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            std::cerr << "d_buf3: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            విడుదల(విభజించు_k_buf);
        }
    }

    విడుదల(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_బఫరు(d_X);
    ggml_vk_destroy_బఫరు(d_Y);
    ggml_vk_destroy_బఫరు(d_D);

    విడుదల(x);
    విడుదల(y);
    విడుదల(d);
}

మూస శూన్య ggml_vk_test_matmul<ggml_fp16_t, దశాంశం>(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t అంకె_it, పూర్ణాంకం విభజించు_k, పూర్ణాంకం shader_కొలత);

స్థిర శూన్య ggml_vk_print_tensor_area(స్థిరరాశి ggml_tensor * tensor, పూర్ణాంకం i0, పూర్ణాంకం i1, పూర్ణాంకం i2, పూర్ణాంకం i3) {
    ఒకవేళ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        తిరిగి;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    కొరకు (పూర్ణాంకం idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    కొరకు (పూర్ణాంకం idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        కొరకు (పూర్ణాంకం idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ఒకవేళ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                దశాంశం val;
                ఒకవేళ (tensor->type == GGML_TYPE_F32) {
                    val = *(దశాంశం *) ((అక్షరం *) tensor->దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } లేకపోతే ఒకవేళ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_గమ్యం_fp32(*(ggml_fp16_t *) ((అక్షరం *) tensor->దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } లేకపోతే {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } లేకపోతే {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

స్థిర శూన్య ggml_vk_quantize_దత్తాంశం(స్థిరరాశి దశాంశం * from, శూన్య * గమ్యం, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, గమ్యం, 0, 1, ne, శూన్యసూచిక);
}

స్థిర శూన్య ggml_vk_dequantize_దత్తాంశం(స్థిరరాశి శూన్య * from, దశాంశం * గమ్యం, size_t ne, ggml_type quant) {
    ఒకవేళ (quant == GGML_TYPE_F32) {
        జ్ఞాపకం_నకలు(గమ్యం, from, పరిమాణం(దశాంశం) * ne);
        తిరిగి;
    }

    స్థిరరాశి స్వయంచాలక * tt = ggml_get_type_traits(quant);

    ggml_గమ్యం_float_t dequant_fn = tt->గమ్యం_float;

    dequant_fn(from, గమ్యం, ne);
}

శూన్య ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    స్థిరరాశి size_t x_sz = పరిమాణం(దశాంశం) * ne;
    స్థిరరాశి size_t x_sz_f16 = పరిమాణం(ggml_fp16_t) * ne;
    స్థిరరాశి size_t qx_sz = ne * ggml_type_కొలత(quant)/ggml_blck_కొలత(quant);
    దశాంశం * x = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(x_sz);
    శూన్య * qx = జ్ఞాపకం_ఇవ్వు(qx_sz);
    vk_బఫరు qx_buf = ggml_vk_create_బఫరు_check(ctx->device, qx_sz, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    vk_బఫరు x_buf = ggml_vk_create_బఫరు_check(ctx->device, x_sz_f16, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    దశాంశం * x_ref = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) జ్ఞాపకం_ఇవ్వు(x_sz_f16);

    కొరకు (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (దశాంశం)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_గమ్యం_fp16(ctx, quant);

    ggml_vk_quantize_దత్తాంశం(x, qx, ne, quant);
    ggml_vk_dequantize_దత్తాంశం(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_బఫరు_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    స్థిరరాశి std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_చివర(subctx);

    స్వయంచాలక begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, నిజం, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_వరుస_command_pools_cleanup(ctx->device);

    స్వయంచాలక end = std::chrono::high_resolution_clock::now();

    ద్విగుణ ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_బఫరు_read(x_buf, 0, x_chk, x_sz_f16);

    పూర్ణాంకం మొదటి_err = -1;

    ద్విగుణ avg_err = 0.0;
    కొరకు (size_t i = 0; i < ne; i++) {
        ద్విగుణ error = std::fabs(x_ref[i] - ggml_fp16_గమ్యం_fp32(x_chk[i]));
        avg_err += error;

        ఒకవేళ (మొదటి_err < 0 && error > 0.05) {
            మొదటి_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::పంక్తి_ముగింపు;

    ఒకవేళ (avg_err > 0.1) {
        std::cerr << "first_error = " << మొదటి_err << std::పంక్తి_ముగింపు;
        std::cerr << "Actual result: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        కొరకు (పూర్ణాంకం i = std::max(0, మొదటి_err - 5); i < std::min((పూర్ణాంకం)ne, మొదటి_err + 5); i++) {
            std::cerr << ggml_fp16_గమ్యం_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::పంక్తి_ముగింపు << "Expected result: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        కొరకు (పూర్ణాంకం i = std::max(0, మొదటి_err - 5); i < std::min((పూర్ణాంకం)ne, మొదటి_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::పంక్తి_ముగింపు;
    }

    ggml_vk_destroy_బఫరు(x_buf);
    ggml_vk_destroy_బఫరు(qx_buf);

    విడుదల(x);
    విడుదల(qx);
    విడుదల(x_ref);
    విడుదల(x_chk);
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

శూన్య ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t అంకె_it, size_t విభజించు_k, size_t shader_కొలత, ggml_type quant, తార్కిక mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << అంకె_it << ", " << విభజించు_k << ", " << ggml_type_name(quant) << ")");
    స్థిరరాశి size_t x_ne = m * k * batch;
    స్థిరరాశి size_t y_ne = k * n * batch;
    స్థిరరాశి size_t d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    తార్కిక f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_కీలకం dq_కీలకం{quant, b_type, అబద్ధం, f16acc};
    స్వయంచాలక dq_it = ctx->device->pipeline_matmul.find(dq_కీలకం);
    ఒకవేళ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.ఖాళీలు()) {
        ఒకవేళ (f16acc) {
            dq_కీలకం.f16acc = అబద్ధం;
            dq_it = ctx->device->pipeline_matmul.find(dq_కీలకం);
        }
    }
    ఒకవేళ (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.ఖాళీలు()) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::పంక్తి_ముగింపు;
        తిరిగి;
    }
    స్వయంచాలక& dq_configs = dq_it->second;
    ఒకవేళ (shader_కొలత >= (పూర్ణాంకం)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_కొలత << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::పంక్తి_ముగింపు;
        తిరిగి;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_కొలత);
    vk_pipeline p = dq_configs[shader_కొలత].aligned ? dq_configs[shader_కొలత].aligned : dq_configs[shader_కొలత].unaligned;

    స్థిరరాశి size_t kpad = mmq ? 0 : ggml_vk_align_కొలత(k, dq_configs[shader_కొలత].align);

    ఒకవేళ (mmq || k != kpad) {
        p = dq_configs[shader_కొలత].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_కొలత);
    }

    ఒకవేళ (p == శూన్యసూచిక) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::పంక్తి_ముగింపు;
        తిరిగి;
    }

    స్థిరరాశి size_t x_sz = పరిమాణం(దశాంశం) * x_ne;
    స్థిరరాశి size_t y_sz = పరిమాణం(దశాంశం) * y_ne;
    స్థిరరాశి size_t qx_sz = x_ne * ggml_type_కొలత(quant)/ggml_blck_కొలత(quant);
    స్థిరరాశి size_t qy_sz = mmq ? y_ne * ggml_type_కొలత(GGML_TYPE_Q8_1)/ggml_blck_కొలత(GGML_TYPE_Q8_1) : y_sz;
    స్థిరరాశి size_t d_sz = పరిమాణం(దశాంశం) * d_ne;
    దశాంశం * x = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(x_sz);
    దశాంశం * y = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(y_sz);
    శూన్య * qx = జ్ఞాపకం_ఇవ్వు(qx_sz);
    vk_బఫరు qx_buf = ggml_vk_create_బఫరు_check(ctx->device, qx_sz, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    vk_బఫరు y_buf = ggml_vk_create_బఫరు_check(ctx->device, y_sz, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    vk_బఫరు qy_buf = ggml_vk_create_బఫరు_check(ctx->device, qy_sz, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    vk_బఫరు d_buf = ggml_vk_create_బఫరు_check(ctx->device, d_sz, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
    దశాంశం * d = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(d_sz);
    దశాంశం * d_chk = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(d_sz);

    కొరకు (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (దశాంశం)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_దత్తాంశం(x, qx, x_ne, quant);

    కొరకు (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (దశాంశం)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    ఒకవేళ (విభజించు_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_విభజించు_k_reduce, అంకె_it);

        ఒకవేళ (ctx->prealloc_విభజించు_k == శూన్యసూచిక || ctx->prealloc_విభజించు_k->size < పరిమాణం(దశాంశం) * d_ne * విభజించు_k) {
            // Resize buffer
            ఒకవేళ (ctx->prealloc_విభజించు_k != శూన్యసూచిక) {
                ggml_vk_destroy_బఫరు(ctx->prealloc_విభజించు_k);
            }
            ctx->prealloc_విభజించు_k = ggml_vk_create_బఫరు_check(ctx->device, పరిమాణం(దశాంశం) * d_ne * విభజించు_k, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
        }
    }
    ఒకవేళ (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, అంకె_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_బఫరు_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_బఫరు_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ఒకవేళ (mmq) {
        కొరకు (size_t i = 0; i < అంకె_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_విభజించు_k, 0, ctx->prealloc_కొలత_విభజించు_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                విభజించు_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_కొలత(GGML_TYPE_Q8_1)) * ggml_type_కొలత(GGML_TYPE_Q8_1), m * పరిమాణం(దశాంశం)
            );
        }
    } లేకపోతే {
        కొరకు (size_t i = 0; i < అంకె_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_విభజించు_k, 0, ctx->prealloc_కొలత_విభజించు_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                విభజించు_k, batch, batch, batch, 1, 1, n,
                k * పరిమాణం(దశాంశం), m * పరిమాణం(దశాంశం)
            );
        }
    }
    ggml_vk_ctx_చివర(subctx);

    స్వయంచాలక begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, నిజం, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_వరుస_command_pools_cleanup(ctx->device);

    స్వయంచాలక end = std::chrono::high_resolution_clock::now();

    ద్విగుణ సమయం_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_బఫరు_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->దత్తాంశం = qx;
    src1_ggml->దత్తాంశం = y;
    tensor_ggml->దత్తాంశం = d_chk;

    ggml_cgraph * cgraph = ggml_new_లేఖాచిత్రం(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_లేఖాచిత్రం_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    ద్విగుణ avg_err = 0.0;
    పూర్ణాంకం మొదటి_err_n = -1;
    పూర్ణాంకం మొదటి_err_m = -1;
    పూర్ణాంకం మొదటి_err_b = -1;

    కొరకు (size_t i = 0; i < m*n*batch; i++) {
        ద్విగుణ err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        ఒకవేళ ((err > 0.05f || std::isnan(err)) && మొదటి_err_n == -1) {
            మొదటి_err_b = i / (m * n);
            మొదటి_err_n = (i % (m * n)) / m;
            మొదటి_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    ద్విగుణ tflops = 2.0*m*n*k*batch*అంకె_it / (సమయం_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    ఒకవేళ (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << విభజించు_k << " matmul " << సమయం_ms / అంకె_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::పంక్తి_ముగింపు;

    ఒకవేళ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << మొదటి_err_m << " n = " << మొదటి_err_n << " b = " << మొదటి_err_b << std::పంక్తి_ముగింపు;
        std::cerr << "Actual result: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        ggml_vk_print_మాతృక_area(d, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);
        std::cerr << std::పంక్తి_ముగింపు;
        std::cerr << "Expected result: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        ggml_vk_print_మాతృక_area(d_chk, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

        std::cerr << "src0: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        ggml_vk_print_మాతృక_area(x, GGML_TYPE_F32, k, m, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);
        std::cerr << std::పంక్తి_ముగింపు;
        std::cerr << "src1: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
        ggml_vk_print_మాతృక_area(y, GGML_TYPE_F32, k, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

        ఒకవేళ (విభజించు_k > 1) {
            దశాంశం * విభజించు_k_buf = (దశాంశం *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(దశాంశం) * d_ne * విభజించు_k);
            ggml_vk_బఫరు_read(ctx->prealloc_విభజించు_k, 0, విభజించు_k_buf, పరిమాణం(దశాంశం) * d_ne * విభజించు_k);

            std::cerr << "d_buf0: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            std::cerr << "d_buf1: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf + d_ne, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            std::cerr << "d_buf2: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            std::cerr << "d_buf3: " << std::పంక్తి_ముగింపు << std::పంక్తి_ముగింపు;
            ggml_vk_print_మాతృక_area(విభజించు_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, మొదటి_err_m, మొదటి_err_n, మొదటి_err_b);

            విడుదల(విభజించు_k_buf);
        }
    }

    ggml_vk_destroy_బఫరు(qx_buf);
    ggml_vk_destroy_బఫరు(y_buf);
    ggml_vk_destroy_బఫరు(qy_buf);
    ggml_vk_destroy_బఫరు(d_buf);

    విడుదల(x);
    విడుదల(qx);
    విడుదల(y);
    విడుదల(d);
    విడుదల(d_chk);
}
#అయితే_ముగింపు

int64_t ggml_vk_get_op_batch_కొలత(స్థిరరాశి ggml_tensor * op) {
    ఎంపిక (op->op) {
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            తిరిగి 0;
        స్థితి GGML_OP_MUL_MAT:
            తిరిగి op->ne[1];
        స్థితి GGML_OP_MUL_MAT_ID:
        స్థితి GGML_OP_ROPE:
        స్థితి GGML_OP_ROPE_వెనుక:
            తిరిగి op->ne[2];
        అప్రమేయ:
            తిరిగి ggml_nrows(op);
    }
}

#నిర్వచితమైతే GGML_VULKAN_CHECK_RESULTS
స్థిర శూన్య ggml_vk_print_లేఖాచిత్రం_origin(స్థిరరాశి ggml_tensor * tensor, std::vector<స్థిరరాశి ggml_tensor *>& పూర్తి, పూర్ణాంకం level = 0) {
    ఒకవేళ (std::find(పూర్తి.begin(), పూర్తి.end(), tensor) != పూర్తి.end() || level > 10) {
        తిరిగి;
    }
    కొరకు (పూర్ణాంకం j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != శూన్యసూచిక) << std::పంక్తి_ముగింపు;

    పూర్తి.push_back(tensor);

    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (tensor->src[i] != శూన్యసూచిక) {
            ggml_vk_print_లేఖాచిత్రం_origin(tensor->src[i], పూర్తి, level + 1);
        }
    }
}

స్థిర శూన్య ggml_vk_print_tensor_area(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, పూర్ణాంకం i0, పూర్ణాంకం i1, పూర్ణాంకం i2, పూర్ణాంకం i3) {
    ఒకవేళ (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        తిరిగి;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    కొరకు (పూర్ణాంకం idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    కొరకు (పూర్ణాంకం idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        కొరకు (పూర్ణాంకం idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            ఒకవేళ (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                దశాంశం val;
                ఒకవేళ (tensor->type == GGML_TYPE_F32) {
                    val = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } లేకపోతే ఒకవేళ (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_గమ్యం_fp32(*(స్థిరరాశి ggml_fp16_t *) ((స్థిరరాశి అక్షరం *) దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } లేకపోతే ఒకవేళ (tensor->type == GGML_TYPE_I32) {
                    val = *(స్థిరరాశి int32_t *) ((స్థిరరాశి అక్షరం *) దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } లేకపోతే {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } లేకపోతే {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

స్థిర శూన్య ggml_vk_print_tensor(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి అక్షరం * name) {
    శూన్య * tensor_దత్తాంశం = tensor->దత్తాంశం;

    స్థిరరాశి తార్కిక is_gpu = tensor->బఫరు != శూన్యసూచిక && ggml_backend_బఫరు_is_vk(tensor->బఫరు);

    ఒకవేళ (is_gpu) {
        స్థిరరాశి size_t tensor_కొలత = ggml_nbytes(tensor);
        tensor_దత్తాంశం = జ్ఞాపకం_ఇవ్వు(tensor_కొలత);

        ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)tensor->బఫరు->context;

        vk_బఫరు బఫరు_gpu = buf_ctx->dev_బఫరు;
        ggml_vk_బఫరు_read(బఫరు_gpu, vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs, tensor_దత్తాంశం, tensor_కొలత);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::పంక్తి_ముగింపు;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::పంక్తి_ముగింపు;
    ఒకవేళ (tensor->src[0] != శూన్యసూచిక) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::పంక్తి_ముగింపు;
    }
    ఒకవేళ (tensor->src[1] != శూన్యసూచిక) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::పంక్తి_ముగింపు;
    }
    std::cerr << std::పంక్తి_ముగింపు << "Result:" << std::పంక్తి_ముగింపు;
    ggml_vk_print_tensor_area(tensor, tensor_దత్తాంశం, 5, 5, 0, 0);
    std::cerr << std::పంక్తి_ముగింపు;
    std::vector<స్థిరరాశి ggml_tensor *> పూర్తి;
    ggml_vk_print_లేఖాచిత్రం_origin(tensor, పూర్తి);

    ఒకవేళ (is_gpu) {
        విడుదల(tensor_దత్తాంశం);
    }
}

శూన్య * comp_ఫలితం;
size_t comp_కొలత;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
శూన్య ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, పూర్ణాంకం tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->అంకె_additional_fused_ops];
    ఒకవేళ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_సమితి_అడ్డువరుసలు) {
        తిరిగి;
    }

    check_counter++;
    ఒకవేళ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        తిరిగి;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    నిర్మాణం ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ అబద్ధం,
    };

    నిర్మాణం ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<నిర్మాణం ggml_tensor *, GGML_MAX_SRC> src_clone = {శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక, శూన్యసూచిక};
    స్థిరరాశి అక్షరం * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<శూన్య *> cloned_mallocs;

    నిర్మాణం ggml_tensor * tensor_clone = శూన్యసూచిక;

    కొరకు (పూర్ణాంకం f = 0; f < ctx->అంకె_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            ఒకవేళ (srci == శూన్యసూచిక) {
                కొనసాగు;
            }
            // If a src tensor has been cloned, use that one
            స్వయంచాలక it = cloned_tensors.find(srci);
            ఒకవేళ (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                కొనసాగు;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            size_t srci_కొలత = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            శూన్య *src_బఫరు = జ్ఞాపకం_ఇవ్వు(srci_కొలత);
            cloned_mallocs.push_back(src_బఫరు);

            srci_clone->దత్తాంశం = src_బఫరు;
            ఒకవేళ (ggml_backend_బఫరు_is_host(srci->బఫరు)) {
                జ్ఞాపకం_నకలు(srci_clone->దత్తాంశం, srci->దత్తాంశం, srci_కొలత);
                జ్ఞాపకం_నకలు(srci_clone->nb, srci->nb, పరిమాణం(size_t) * GGML_MAX_DIMS);
            } లేకపోతే ఒకవేళ (ggml_backend_బఫరు_is_vk(srci->బఫరు)) {
                ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)srci->బఫరు->context;
                vk_బఫరు& బఫరు_gpu = buf_ctx->dev_బఫరు;
                uint64_t స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(srci) + srci->view_offs;
                ఒకవేళ (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    కొరకు (పూర్ణాంకం i3 = 0; i3 < srci->ne[3]; i3++) {
                        కొరకు (పూర్ణాంకం i2 = 0; i2 < srci->ne[2]; i2++) {
                            స్థిరరాశి పూర్ణాంకం idx = i3*srci->ne[2] + i2;
                            ggml_vk_బఫరు_read(బఫరు_gpu, స్థానభ్రంశం + idx * srci->nb[2], ((అక్షరం *)srci_clone->దత్తాంశం + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    కొరకు (పూర్ణాంకం i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } లేకపోతే {
                    ఒకవేళ (స్థానభ్రంశం + srci_కొలత >= బఫరు_gpu->size) {
                        srci_కొలత = బఫరు_gpu->size - స్థానభ్రంశం;
                    }
                    ggml_vk_బఫరు_read(బఫరు_gpu, స్థానభ్రంశం, srci_clone->దత్తాంశం, srci_కొలత);
                    జ్ఞాపకం_నకలు(srci_clone->nb, srci->nb, పరిమాణం(size_t) * GGML_MAX_DIMS);
                }
            } లేకపోతే {
                GGML_ABORT("fatal error");
            }

            ఒకవేళ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        ఒకవేళ (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            స్థిరరాశి దశాంశం * params = (స్థిరరాశి దశాంశం *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            ఒకవేళ (src_clone[4]) {
                ggml_flash_attn_ext_కూడు_sinks(tensor_clone, src_clone[4]);
            }
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(పూర్ణాంకం *)tensor->op_params);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_బహుళకం) tensor->op_params[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SCALE) {
            స్థిరరాశి దశాంశం * params = (స్థిరరాశి దశాంశం *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_ARANGE) {
            స్థిరరాశి దశాంశం start = ggml_get_op_params_f32(tensor, 0);
            స్థిరరాశి దశాంశం stop = ggml_get_op_params_f32(tensor, 1);
            స్థిరరాశి దశాంశం అడుగు = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, అడుగు);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_FILL) {
            స్థిరరాశి దశాంశం value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CLAMP) {
            స్థిరరాశి దశాంశం * params = (స్థిరరాశి దశాంశం *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_REPEAT_వెనుక) {
            tensor_clone = ggml_repeat_వెనుక(ggml_ctx, src_clone[0], tensor);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_కూడు) {
            tensor_clone = ggml_కూడు(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_సమితి) {
            tensor_clone = ggml_సమితి(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(దశాంశం *)tensor->op_params);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_సమూహం_NORM) {
            స్థిరరాశి దశాంశం * float_params = (స్థిరరాశి దశాంశం *)tensor->op_params;
            tensor_clone = ggml_సమూహం_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(దశాంశం *)tensor->op_params);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_RMS_NORM_వెనుక) {
            స్థిరరాశి దశాంశం eps = ((దశాంశం *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_వెనుక(ggml_ctx, src_clone[0], src_clone[1], eps);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SILU_వెనుక) {
            tensor_clone = ggml_silu_వెనుక(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_L2_NORM) {
            స్థిరరాశి దశాంశం eps = ((దశాంశం *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SOFT_MAX) {
            ఒకవేళ (tensor->src[1] != శూన్యసూచిక) {
                స్థిరరాశి దశాంశం * params = (స్థిరరాశి దశాంశం *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } లేకపోతే {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SOFT_MAX_వెనుక) {
            tensor_clone = ggml_soft_max_ext_వెనుక(ggml_ctx, src_clone[0], src_clone[1], ((దశాంశం *)tensor->op_params)[0], ((దశాంశం *)tensor->op_params)[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_DIAG_ముసుగు_INF) {
            tensor_clone = ggml_diag_ముసుగు_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_వెనుక) {
            స్థిరరాశి పూర్ణాంకం n_dims      = ((int32_t *) tensor->op_params)[1];
            స్థిరరాశి పూర్ణాంకం mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            స్థిరరాశి పూర్ణాంకం n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            స్థిరరాశి దశాంశం freq_పునాది       = ((దశాంశం *) tensor->op_params)[5];
            స్థిరరాశి దశాంశం freq_scale      = ((దశాంశం *) tensor->op_params)[6];
            స్థిరరాశి దశాంశం ext_గుణకం      = ((దశాంశం *) tensor->op_params)[7];
            స్థిరరాశి దశాంశం attn_గుణకం     = ((దశాంశం *) tensor->op_params)[8];
            స్థిరరాశి దశాంశం beta_fast       = ((దశాంశం *) tensor->op_params)[9];
            స్థిరరాశి దశాంశం beta_slow       = ((దశాంశం *) tensor->op_params)[10];
            ఒకవేళ (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                ఒకవేళ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
                } లేకపోతే {
                    tensor_clone = ggml_rope_multi_వెనుక(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
                }
            } లేకపోతే {
                ఒకవేళ (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
                } లేకపోతే {
                    tensor_clone = ggml_rope_ext_వెనుక(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
                }
            }
            స్థిరరాశి పూర్ణాంకం n_offs = ((int32_t *) tensor->op_params)[15];
            ఒకవేళ (n_offs != 0) {
                tensor_clone = ggml_rope_సమితి_స్థానభ్రంశం(tensor_clone, n_offs);
            }
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_UNARY) {
            ఎంపిక (ggml_get_unary_op(tensor)) {
            స్థితి GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                tensor_clone = ggml_gelu_శీఘ్ర(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_సమితి_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_సమితి_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_సమితి_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_సమితి_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                ఆపు;
            స్థితి GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_అడుగు:
                tensor_clone = ggml_అడుగు(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_విడత:
                tensor_clone = ggml_విడత(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                ఆపు;
            స్థితి GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                ఆపు;
            అప్రమేయ:
                std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::పంక్తి_ముగింపు;
                GGML_ABORT("fatal error");
            }
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_GLU) {
            ఒకవేళ (src_clone[1] == శూన్యసూచిక) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } లేకపోతే {
                tensor_clone = ggml_glu_విభజించు(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_సమితి_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_సమితి_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            ఒకవేళ (tensor->src[1] == శూన్యసూచిక) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } లేకపోతే {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_క్రమమార్పు) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_క్రమమార్పు(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_GET_అడ్డువరుసలు) {
            tensor_clone = ggml_get_అడ్డువరుసలు(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_క్రమీకరించు_క్రమం) *(పూర్ణాంకం *)tensor->op_params);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_శిఖరం_K) {
            tensor_clone = ggml_శిఖరం_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_కూడిక) {
            tensor_clone = ggml_కూడిక(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_కూడిక_అడ్డువరుసలు) {
            tensor_clone = ggml_కూడిక_అడ్డువరుసలు(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_DSV4_HC_PRE) {
            ఒకవేళ (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } లేకపోతే {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_సగటు) {
            tensor_clone = ggml_సగటు(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_వెనుక) {
            tensor_clone = ggml_cross_entropy_loss_వెనుక(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_లెక్క_EQUAL) {
            tensor_clone = ggml_లెక్క_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], నిజం, నిజం, అబద్ధం);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_IM2_నిలువు) {
            స్థిరరాశి int32_t s0 = tensor->op_params[0];
            స్థిరరాశి int32_t s1 = tensor->op_params[1];
            స్థిరరాశి int32_t p0 = tensor->op_params[2];
            స్థిరరాశి int32_t p1 = tensor->op_params[3];
            స్థిరరాశి int32_t d0 = tensor->op_params[4];
            స్థిరరాశి int32_t d1 = tensor->op_params[5];

            స్థిరరాశి తార్కిక is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_IM2_నిలువు_3_D) {
            స్థిరరాశి int32_t s0 = tensor->op_params[0];
            స్థిరరాశి int32_t s1 = tensor->op_params[1];
            స్థిరరాశి int32_t s2 = tensor->op_params[2];
            స్థిరరాశి int32_t p0 = tensor->op_params[3];
            స్థిరరాశి int32_t p1 = tensor->op_params[4];
            స్థిరరాశి int32_t p2 = tensor->op_params[5];
            స్థిరరాశి int32_t d0 = tensor->op_params[6];
            స్థిరరాశి int32_t d1 = tensor->op_params[7];
            స్థిరరాశి int32_t d2 = tensor->op_params[8];
            స్థిరరాశి int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            స్థిరరాశి int32_t dim = tensor->op_params[0];
            స్థిరరాశి int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            స్థిరరాశి int32_t s0 = tensor->op_params[0];
            స్థిరరాశి int32_t p0 = tensor->op_params[1];
            స్థిరరాశి int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_COL2IM_1D) {
            స్థిరరాశి int32_t stride = tensor->op_params[0];
            స్థిరరాశి int32_t oc     = tensor->op_params[1];
            స్థిరరాశి int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_POOL_1D) {
            గణన ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            స్థిరరాశి int32_t k0 = tensor->op_params[1];
            స్థిరరాశి int32_t s0 = tensor->op_params[2];
            స్థిరరాశి int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_POOL_2D) {
            గణన ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            స్థిరరాశి int32_t k0 = tensor->op_params[1];
            స్థిరరాశి int32_t k1 = tensor->op_params[2];
            స్థిరరాశి int32_t s0 = tensor->op_params[3];
            స్థిరరాశి int32_t s1 = tensor->op_params[4];
            స్థిరరాశి int32_t p0 = tensor->op_params[5];
            స్థిరరాశి int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CONV_2D) {
            స్థిరరాశి int32_t s0 = tensor->op_params[0];
            స్థిరరాశి int32_t s1 = tensor->op_params[1];
            స్థిరరాశి int32_t p0 = tensor->op_params[2];
            స్థిరరాశి int32_t p1 = tensor->op_params[3];
            స్థిరరాశి int32_t d0 = tensor->op_params[4];
            స్థిరరాశి int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CONV_3D) {
            స్థిరరాశి int32_t s0 = tensor->op_params[0];
            స్థిరరాశి int32_t s1 = tensor->op_params[1];
            స్థిరరాశి int32_t s2 = tensor->op_params[2];
            స్థిరరాశి int32_t p0 = tensor->op_params[3];
            స్థిరరాశి int32_t p1 = tensor->op_params[4];
            స్థిరరాశి int32_t p2 = tensor->op_params[5];
            స్థిరరాశి int32_t d0 = tensor->op_params[6];
            స్థిరరాశి int32_t d1 = tensor->op_params[7];
            స్థిరరాశి int32_t d2 = tensor->op_params[8];
            స్థిరరాశి int32_t IC = tensor->op_params[9];
            స్థిరరాశి int32_t N  = tensor->op_params[10];
            స్థిరరాశి int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_ప్రత్యక్ష(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CONV_2D_DW) {
            స్థిరరాశి int32_t s0 = tensor->op_params[0];
            స్థిరరాశి int32_t s1 = tensor->op_params[1];
            స్థిరరాశి int32_t p0 = tensor->op_params[2];
            స్థిరరాశి int32_t p1 = tensor->op_params[3];
            స్థిరరాశి int32_t d0 = tensor->op_params[4];
            స్థిరరాశి int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_ప్రత్యక్ష(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            స్థిరరాశి int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_LEAKY_RELU) {
            స్థిరరాశి దశాంశం * op_params = (స్థిరరాశి దశాంశం *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], అబద్ధం);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_GATED_రేఖీయ_ATTN) {
            స్థిరరాశి దశాంశం * op_params = (స్థిరరాశి దశాంశం *)tensor->op_params;
            tensor_clone = ggml_gated_రేఖీయ_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_OPT_అడుగు_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_అడుగు_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_OPT_అడుగు_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_అడుగు_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_కూడు_ID) {
            tensor_clone = ggml_కూడు_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SSM_పరిశీలన) {
            స్థిరరాశి int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_పరిశీలన(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } లేకపోతే ఒకవేళ (tensor->op == GGML_OP_ROLL) {
            స్థిరరాశి int32_t s0 = tensor->op_params[0];
            స్థిరరాశి int32_t s1 = tensor->op_params[1];
            స్థిరరాశి int32_t s2 = tensor->op_params[2];
            స్థిరరాశి int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        లేకపోతే {
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::పంక్తి_ముగింపు;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_లేఖాచిత్రం(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_లేఖాచిత్రం_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    ఒకవేళ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_కొలత = ggml_nbytes(tensor_clone);

    comp_ఫలితం = జ్ఞాపకం_ఇవ్వు(comp_కొలత);
    జ్ఞాపకం_నకలు(comp_ఫలితం, tensor_clone->దత్తాంశం, comp_కొలత);
    జ్ఞాపకం_నకలు(comp_nb, tensor_clone->nb, పరిమాణం(size_t) * GGML_MAX_DIMS);

    కొరకు (స్వయంచాలక m : cloned_mallocs) {
        విడుదల(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

శూన్య ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, పూర్ణాంకం tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->అంకె_additional_fused_ops];
    ఒకవేళ (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_సమితి_అడ్డువరుసలు) {
        తిరిగి;
    }

    ఒకవేళ (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        తిరిగి;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    శూన్య * tensor_దత్తాంశం = tensor->దత్తాంశం;

    ఒకవేళ (ggml_backend_బఫరు_is_vk(tensor->బఫరు)) {
        size_t tensor_కొలత = ggml_nbytes(tensor);
        tensor_దత్తాంశం = జ్ఞాపకం_ఇవ్వు(tensor_కొలత);

        ggml_backend_vk_బఫరు_context * buf_ctx = (ggml_backend_vk_బఫరు_context *)tensor->బఫరు->context;

        vk_బఫరు& బఫరు_gpu = buf_ctx->dev_బఫరు;
        uint64_t స్థానభ్రంశం = vk_tensor_స్థానభ్రంశం(tensor) + tensor->view_offs;
        ఒకవేళ (స్థానభ్రంశం + tensor_కొలత >= బఫరు_gpu->size) {
            tensor_కొలత = బఫరు_gpu->size - స్థానభ్రంశం;
        }

        ggml_vk_బఫరు_read(బఫరు_gpu, స్థానభ్రంశం, tensor_దత్తాంశం, tensor_కొలత);
    }

    దశాంశం మొదటి_error_ఫలితం = -1.0f;
    దశాంశం మొదటి_error_correct = -1.0f;
    std::array<పూర్ణాంకం, 4> మొదటి_error = { -1, -1, -1, -1 };
    ద్విగుణ avg_err = 0.0;
    size_t counter = 0;

    కొరకు (పూర్ణాంకం i3 = 0; i3 < tensor->ne[3]; i3++) {
        కొరకు (పూర్ణాంకం i2 = 0; i2 < tensor->ne[2]; i2++) {
            కొరకు (పూర్ణాంకం i1 = 0; i1 < tensor->ne[1]; i1++) {
                కొరకు (పూర్ణాంకం i0 = 0; i0 < tensor->ne[0]; i0++) {
                    స్థిరరాశి తార్కిక బఫరు_కొలత_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_కొలత;
                    దశాంశం correct = 0.0f;
                    దశాంశం ఫలితం = 0.0f;

                    ఒకవేళ (బఫరు_కొలత_fit) {
                        ఒకవేళ (tensor->type == GGML_TYPE_F32) {
                            correct = *(దశాంశం *) ((అక్షరం *) comp_ఫలితం + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ఫలితం  = *(దశాంశం *) ((అక్షరం *) tensor_దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } లేకపోతే ఒకవేళ (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_గమ్యం_fp32(*(ggml_fp16_t *) ((అక్షరం *) comp_ఫలితం + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ఫలితం  = ggml_fp16_గమ్యం_fp32(*(ggml_fp16_t *) ((అక్షరం *) tensor_దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } లేకపోతే ఒకవేళ (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_గమ్యం_fp32(*(ggml_bf16_t *) ((అక్షరం *) comp_ఫలితం + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            ఫలితం  = ggml_bf16_గమ్యం_fp32(*(ggml_bf16_t *) ((అక్షరం *) tensor_దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } లేకపోతే ఒకవేళ (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((అక్షరం *) comp_ఫలితం + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ఫలితం  = *(int32_t *) ((అక్షరం *) tensor_దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } లేకపోతే ఒకవేళ (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((అక్షరం *) comp_ఫలితం + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            ఫలితం  = *(int64_t *) ((అక్షరం *) tensor_దత్తాంశం + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } లేకపోతే {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::పంక్తి_ముగింపు;
                        }
                    } లేకపోతే {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::పంక్తి_ముగింపు;
                        GGML_ABORT("fatal error");
                    }

                    ఒకవేళ ((std::isnan(correct) != std::isnan(ఫలితం)) || (std::isinf(correct) != std::isinf(ఫలితం)) || !బఫరు_కొలత_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << ఫలితం << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::పంక్తి_ముగింపు;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::పంక్తి_ముగింపు;
                        ఒకవేళ (src0 != శూన్యసూచిక) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::పంక్తి_ముగింపు;
                        }
                        ఒకవేళ (src1 != శూన్యసూచిక) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::పంక్తి_ముగింపు;
                        }
                        ఒకవేళ (src2 != శూన్యసూచిక) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::పంక్తి_ముగింపు;
                        }
                        ఒకవేళ (src3 != శూన్యసూచిక) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::పంక్తి_ముగింపు;
                        }
                        std::cerr << "First error: result=" << మొదటి_error_ఫలితం << " correct=" << మొదటి_error_correct  << " i3=" << మొదటి_error[3] << " i2=" << మొదటి_error[2] << " i1=" << మొదటి_error[1] << " i0=" << మొదటి_error[0] << std::పంక్తి_ముగింపు;
                        std::cerr << std::పంక్తి_ముగింపు << "Result:" << std::పంక్తి_ముగింపు;
                        ggml_vk_print_tensor_area(tensor, tensor_దత్తాంశం, i0, i1, i2, i3);
                        std::cerr << std::పంక్తి_ముగింపు << "Correct:" << std::పంక్తి_ముగింపు;
                        ggml_vk_print_tensor_area(tensor, comp_ఫలితం, i0, i1, i2, i3);
                        std::cerr << std::పంక్తి_ముగింపు;
                        std::vector<స్థిరరాశి ggml_tensor *> పూర్తి;
                        ggml_vk_print_లేఖాచిత్రం_origin(tensor, పూర్తి);
                        GGML_ABORT("fatal error");
                    }
                    స్థిరరాశి ద్విగుణ denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    ఒకవేళ (మొదటి_error[0] == -1 && std::fabs(correct - ఫలితం) / denom > 0.5) {
                        మొదటి_error[0] = i0;
                        మొదటి_error[1] = i1;
                        మొదటి_error[2] = i2;
                        మొదటి_error[3] = i3;
                        మొదటి_error_ఫలితం = ఫలితం;
                        మొదటి_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    ఒకవేళ (!std::isinf(correct) && !std::isinf(ఫలితం) && !std::isnan(correct) && !std::isnan(ఫలితం)) {
                        avg_err += std::fabs(correct - ఫలితం) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    ఒకవేళ (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::పంక్తి_ముగింపు;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::పంక్తి_ముగింపు;
        ఒకవేళ (src0 != శూన్యసూచిక) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::పంక్తి_ముగింపు;
        }
        ఒకవేళ (src1 != శూన్యసూచిక) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::పంక్తి_ముగింపు;
        }
        ఒకవేళ (src2 != శూన్యసూచిక) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::పంక్తి_ముగింపు;
        }
        ఒకవేళ (src3 != శూన్యసూచిక) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::పంక్తి_ముగింపు;
        }
        std::cerr << "First error: result=" << మొదటి_error_ఫలితం << " correct=" << మొదటి_error_correct  << " i3=" << మొదటి_error[3] << " i2=" << మొదటి_error[2] << " i1=" << మొదటి_error[1] << " i0=" << మొదటి_error[0] << std::పంక్తి_ముగింపు;
        std::cerr << std::పంక్తి_ముగింపు << "Result:" << std::పంక్తి_ముగింపు;
        ggml_vk_print_tensor_area(tensor, tensor_దత్తాంశం, 5, 5, 0, 0);
        std::cerr << std::పంక్తి_ముగింపు << "Correct:" << std::పంక్తి_ముగింపు;
        ggml_vk_print_tensor_area(tensor, comp_ఫలితం, 5, 5, 0, 0);
        std::cerr << std::పంక్తి_ముగింపు;
        std::vector<స్థిరరాశి ggml_tensor *> పూర్తి;
        ggml_vk_print_లేఖాచిత్రం_origin(tensor, పూర్తి);
    }

    ఒకవేళ (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::పంక్తి_ముగింపు;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::పంక్తి_ముగింపు;
        ఒకవేళ (src0 != శూన్యసూచిక) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::పంక్తి_ముగింపు;
        }
        ఒకవేళ (src1 != శూన్యసూచిక) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::పంక్తి_ముగింపు;
        }
        ఒకవేళ (src2 != శూన్యసూచిక) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::పంక్తి_ముగింపు;
        }
        ఒకవేళ (src3 != శూన్యసూచిక) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::పంక్తి_ముగింపు;
        }
        std::cerr << "First error: result=" << మొదటి_error_ఫలితం << " correct=" << మొదటి_error_correct  << " i3=" << మొదటి_error[3] << " i2=" << మొదటి_error[2] << " i1=" << మొదటి_error[1] << " i0=" << మొదటి_error[0] << std::పంక్తి_ముగింపు;
        std::cerr << std::పంక్తి_ముగింపు << "Result:" << std::పంక్తి_ముగింపు;
        ggml_vk_print_tensor_area(tensor, tensor_దత్తాంశం, మొదటి_error[0], మొదటి_error[1], మొదటి_error[2], మొదటి_error[3]);
        std::cerr << std::పంక్తి_ముగింపు << "Correct:" << std::పంక్తి_ముగింపు;
        ggml_vk_print_tensor_area(tensor, comp_ఫలితం, మొదటి_error[0], మొదటి_error[1], మొదటి_error[2], మొదటి_error[3]);
        std::cerr << std::పంక్తి_ముగింపు;
        std::vector<స్థిరరాశి ggml_tensor *> పూర్తి;
        ggml_vk_print_లేఖాచిత్రం_origin(tensor, పూర్తి);
        GGML_ABORT("fatal error");
    } లేకపోతే {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::పంక్తి_ముగింపు;
    }

    విడుదల(comp_ఫలితం);
    comp_ఫలితం = శూన్యసూచిక;
    comp_కొలత = 0;

    ఒకవేళ (ggml_backend_బఫరు_is_vk(tensor->బఫరు)) {
        విడుదల(tensor_దత్తాంశం);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#అయితే_ముగింపు

