#ضمن "ggml-vulkan-common.h"

منطقي vk_يادگيري_logger_enabled = خطأ;

منطقي vk_perf_logger_enabled = خطأ;

منطقي vk_perf_logger_concurrent = خطأ;

منطقي vk_enable_sync_logger = خطأ;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

فراغ vk_يادگيري_logger::log_ورهاست(vk_بفر_ref buf_ref, size_t size) {
    إذا (!vk_يادگيري_logger_enabled) {
        أرجع;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_بفر buf = buf_ref.تالو();
    ثابت منطقي device = منطقي(buf->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eDeviceLocal);
    ثابت std::string type = device ? "device" : "host";
    allocations[buf->بفر] = size;
    ڪل_device += device ? size : 0;
    ڪل_host += device ? 0 : size;
    VK_LOG_يادگيري(buf->device->name << ": +" << format_سائيز(size) << " " << type << " at " << buf->بفر << ". Total device: " << format_سائيز(ڪل_device) << ", total host: " << format_سائيز(ڪل_host));
}

فراغ vk_يادگيري_logger::log_deallocation(vk_بفر_ref buf_ref) {
    إذا (buf_ref.expired() || buf_ref.تالو()->size == 0 || !vk_يادگيري_logger_enabled) {
        أرجع;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_بفر buf = buf_ref.تالو();
    ثابت منطقي device = منطقي(buf->يادگيري_property_flags & vk::يادگيري_Property_Flag_بٽ::eDeviceLocal);
    std::string type = device ? "device" : "host";
    خودڪار it = allocations.find(buf->بفر);
    إذا (it != allocations.end()) {
        ڪل_device -= device ? it->second : 0;
        ڪل_host -= device ? 0 : it->second;
        VK_LOG_يادگيري(buf->device->name << ": -" << format_سائيز(it->second) << " " << type << " at " << buf->بفر << ". Total device: " << format_سائيز(ڪل_device) << ", total host: " << format_سائيز(ڪل_host));
        allocations.erase(it);
    } وإلا {
        VK_LOG_يادگيري("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->بفر);
    }
}

#جيڪڏهن_وصف_ٿيل GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

ساكن فراغ ggml_vk_print_tensor(ثابت ggml_tensor * tensor, ثابت حرف * name);
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_VULKAN_لاڳيتو_TESTS
ساكن فراغ ggml_vk_print_ميٽرڪس_area(ثابت فراغ * معلومات, ggml_type type, صحيح ne0, صحيح ne1, صحيح i0, صحيح i1, صحيح i2) {
    إذا (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        أرجع;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    لكل (صحيح idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    لكل (صحيح idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        لكل (صحيح idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            إذا (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                عائم val;
                إذا (type == GGML_TYPE_F32) {
                    val = *((ثابت عائم *) معلومات + i2*ne1*ne0 + idx1*ne0 + idx0);
                } وإلا إذا (type == GGML_TYPE_F16) {
                    val = ggml_fp16_منزل_fp32(*((ثابت ggml_fp16_t *) معلومات + i2*ne1*ne0 + idx1*ne0 + idx0));
                } وإلا {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } وإلا {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

سانچو <قسم_نالو X_TYPE, قسم_نالو Y_TYPE>
فراغ ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t انگ_it, صحيح ورهايو_k, صحيح shader_سائيز) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << انگ_it << ", " << ورهايو_k << ", " << shader_سائيز << ")");
    ثابت size_t x_ne = m * k * batch;
    ثابت size_t y_ne = k * n * batch;
    ثابت size_t d_ne = m * n * batch;

    ggml_type x_type = std::is_same<عائم, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<عائم, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_ڪنجي mm_test_ڪنجي{x_type, y_type, خطأ, خطأ};
    خودڪار mm_test_it = ctx->device->pipeline_matmul.find(mm_test_ڪنجي);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.خالي_جايون());
    خودڪار& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_سائيز >= 0 && shader_سائيز < (صحيح)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_سائيز);
    vk_pipeline p = mm_test_configs[shader_سائيز].aligned ? mm_test_configs[shader_سائيز].aligned : mm_test_configs[shader_سائيز].unaligned;

    ثابت size_t kpad = ggml_vk_align_سائيز(k, mm_test_configs[shader_سائيز].align);

    إذا (k != kpad) {
        p = mm_test_configs[shader_سائيز].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_سائيز);
    }

    إذا (ورهايو_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ورهايو_k_reduce, انگ_it);

        إذا (ctx->prealloc_ورهايو_k == خالي_اشارو || ctx->prealloc_ورهايو_k->size < حجم(عائم) * d_ne * ورهايو_k) {
            // Resize buffer
            إذا (ctx->prealloc_ورهايو_k != خالي_اشارو) {
                ggml_vk_destroy_بفر(ctx->prealloc_ورهايو_k);
            }
            ctx->prealloc_ورهايو_k = ggml_vk_create_بفر_check(ctx->device, حجم(عائم) * d_ne * ورهايو_k, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_بفر d_X = ggml_vk_create_بفر_check(ctx->device, حجم(X_TYPE) * x_ne, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    vk_بفر d_Y = ggml_vk_create_بفر_check(ctx->device, حجم(Y_TYPE) * y_ne, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    vk_بفر d_D = ggml_vk_create_بفر_check(ctx->device, حجم(عائم) * d_ne, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) يادگيري_ڏيو(حجم(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) يادگيري_ڏيو(حجم(Y_TYPE) * y_ne);
    عائم* d = (عائم *) يادگيري_ڏيو(حجم(عائم) * d_ne);

    لكل (size_t i = 0; i < x_ne; i++) {
        إذا (std::is_same<عائم, X_TYPE>()) {
            x[i] = (rand() / (عائم)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } وإلا إذا (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_منزل_fp16((rand() / (عائم)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } وإلا {
            GGML_ABORT("fatal error");
        }
    }
    لكل (size_t i = 0; i < y_ne; i++) {
        إذا (std::is_same<عائم, Y_TYPE>()) {
            y[i] = (rand() / (عائم)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } وإلا إذا (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_منزل_fp16((rand() / (عائم)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } وإلا {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_بفر_write(d_X, 0, x, حجم(X_TYPE) * k * m * batch);
    ggml_vk_بفر_write(d_Y, 0, y, حجم(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    لكل (size_t i = 0; i < انگ_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_ورهايو_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            ورهايو_k, batch, batch, batch, 1, 1, n,
            k * حجم(Y_TYPE), m * حجم(عائم)
        );
    }
    ggml_vk_ctx_پڇاڙي(subctx);

    خودڪار begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, سچ, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_قطار_command_pools_cleanup(ctx->device);

    خودڪار end = std::chrono::high_resolution_clock::now();
    مزدوج وقت = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_بفر_read(d_D, 0, d, حجم(عائم) * d_ne);

    عائم * d_chk = (عائم *) يادگيري_ڏيو(حجم(عائم) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    إذا (std::is_same<عائم, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } وإلا إذا (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } وإلا {
        GGML_ABORT("fatal error");
    }
    إذا (std::is_same<عائم, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } وإلا إذا (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } وإلا {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->معلومات = x;
    src1_ggml->معلومات = y;
    tensor_ggml->معلومات = d_chk;

    ggml_cgraph * cgraph = ggml_new_گراف(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_گراف_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    مزدوج avg_err = 0.0;
    صحيح پهريون_err_n = -1;
    صحيح پهريون_err_m = -1;
    صحيح پهريون_err_b = -1;

    لكل (size_t i = 0; i < m*n*batch; i++) {
        مزدوج err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        إذا ((err > 0.05f || std::isnan(err)) && پهريون_err_n == -1) {
            پهريون_err_b = i / (m * n);
            پهريون_err_n = (i % (m * n)) / m;
            پهريون_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    مزدوج tflops = 2.0*m*n*k*batch*انگ_it / (وقت / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << ورهايو_k << " matmul " << وقت / انگ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::سٽ_پڄاڻي;

    إذا (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << پهريون_err_m << " n = " << پهريون_err_n << " b = " << پهريون_err_b << std::سٽ_پڄاڻي;
        std::cerr << "Actual result: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        ggml_vk_print_ميٽرڪس_area(d, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);
        std::cerr << "Expected result: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        ggml_vk_print_ميٽرڪس_area(d_chk, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

        إذا (ورهايو_k > 1) {
            عائم * ورهايو_k_buf = (عائم *) يادگيري_ڏيو(حجم(عائم) * d_ne * ورهايو_k);
            ggml_vk_بفر_read(ctx->prealloc_ورهايو_k, 0, ورهايو_k_buf, حجم(عائم) * d_ne * ورهايو_k);

            std::cerr << "d_buf0: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            std::cerr << "d_buf1: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf + d_ne, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            std::cerr << "d_buf2: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            std::cerr << "d_buf3: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            حرر(ورهايو_k_buf);
        }
    }

    حرر(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_بفر(d_X);
    ggml_vk_destroy_بفر(d_Y);
    ggml_vk_destroy_بفر(d_D);

    حرر(x);
    حرر(y);
    حرر(d);
}

سانچو فراغ ggml_vk_test_matmul<ggml_fp16_t, عائم>(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t انگ_it, صحيح ورهايو_k, صحيح shader_سائيز);

ساكن فراغ ggml_vk_print_tensor_area(ثابت ggml_tensor * tensor, صحيح i0, صحيح i1, صحيح i2, صحيح i3) {
    إذا (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        أرجع;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    لكل (صحيح idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    لكل (صحيح idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        لكل (صحيح idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            إذا (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                عائم val;
                إذا (tensor->type == GGML_TYPE_F32) {
                    val = *(عائم *) ((حرف *) tensor->معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } وإلا إذا (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_منزل_fp32(*(ggml_fp16_t *) ((حرف *) tensor->معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } وإلا {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } وإلا {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ساكن فراغ ggml_vk_quantize_معلومات(ثابت عائم * from, فراغ * منزل, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, منزل, 0, 1, ne, خالي_اشارو);
}

ساكن فراغ ggml_vk_dequantize_معلومات(ثابت فراغ * from, عائم * منزل, size_t ne, ggml_type quant) {
    إذا (quant == GGML_TYPE_F32) {
        يادگيري_نقل(منزل, from, حجم(عائم) * ne);
        أرجع;
    }

    ثابت خودڪار * tt = ggml_get_type_traits(quant);

    ggml_منزل_float_t dequant_fn = tt->منزل_float;

    dequant_fn(from, منزل, ne);
}

فراغ ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    ثابت size_t x_sz = حجم(عائم) * ne;
    ثابت size_t x_sz_f16 = حجم(ggml_fp16_t) * ne;
    ثابت size_t qx_sz = ne * ggml_type_سائيز(quant)/ggml_blck_سائيز(quant);
    عائم * x = (عائم *) يادگيري_ڏيو(x_sz);
    فراغ * qx = يادگيري_ڏيو(qx_sz);
    vk_بفر qx_buf = ggml_vk_create_بفر_check(ctx->device, qx_sz, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    vk_بفر x_buf = ggml_vk_create_بفر_check(ctx->device, x_sz_f16, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    عائم * x_ref = (عائم *) يادگيري_ڏيو(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) يادگيري_ڏيو(x_sz_f16);

    لكل (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (عائم)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_منزل_fp16(ctx, quant);

    ggml_vk_quantize_معلومات(x, qx, ne, quant);
    ggml_vk_dequantize_معلومات(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_بفر_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    ثابت std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_پڇاڙي(subctx);

    خودڪار begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, سچ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_قطار_command_pools_cleanup(ctx->device);

    خودڪار end = std::chrono::high_resolution_clock::now();

    مزدوج ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_بفر_read(x_buf, 0, x_chk, x_sz_f16);

    صحيح پهريون_err = -1;

    مزدوج avg_err = 0.0;
    لكل (size_t i = 0; i < ne; i++) {
        مزدوج error = std::fabs(x_ref[i] - ggml_fp16_منزل_fp32(x_chk[i]));
        avg_err += error;

        إذا (پهريون_err < 0 && error > 0.05) {
            پهريون_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::سٽ_پڄاڻي;

    إذا (avg_err > 0.1) {
        std::cerr << "first_error = " << پهريون_err << std::سٽ_پڄاڻي;
        std::cerr << "Actual result: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        لكل (صحيح i = std::max(0, پهريون_err - 5); i < std::min((صحيح)ne, پهريون_err + 5); i++) {
            std::cerr << ggml_fp16_منزل_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::سٽ_پڄاڻي << "Expected result: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        لكل (صحيح i = std::max(0, پهريون_err - 5); i < std::min((صحيح)ne, پهريون_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::سٽ_پڄاڻي;
    }

    ggml_vk_destroy_بفر(x_buf);
    ggml_vk_destroy_بفر(qx_buf);

    حرر(x);
    حرر(qx);
    حرر(x_ref);
    حرر(x_chk);
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

فراغ ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t انگ_it, size_t ورهايو_k, size_t shader_سائيز, ggml_type quant, منطقي mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << انگ_it << ", " << ورهايو_k << ", " << ggml_type_name(quant) << ")");
    ثابت size_t x_ne = m * k * batch;
    ثابت size_t y_ne = k * n * batch;
    ثابت size_t d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    منطقي f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_ڪنجي dq_ڪنجي{quant, b_type, خطأ, f16acc};
    خودڪار dq_it = ctx->device->pipeline_matmul.find(dq_ڪنجي);
    إذا (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.خالي_جايون()) {
        إذا (f16acc) {
            dq_ڪنجي.f16acc = خطأ;
            dq_it = ctx->device->pipeline_matmul.find(dq_ڪنجي);
        }
    }
    إذا (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.خالي_جايون()) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::سٽ_پڄاڻي;
        أرجع;
    }
    خودڪار& dq_configs = dq_it->second;
    إذا (shader_سائيز >= (صحيح)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_سائيز << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::سٽ_پڄاڻي;
        أرجع;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_سائيز);
    vk_pipeline p = dq_configs[shader_سائيز].aligned ? dq_configs[shader_سائيز].aligned : dq_configs[shader_سائيز].unaligned;

    ثابت size_t kpad = mmq ? 0 : ggml_vk_align_سائيز(k, dq_configs[shader_سائيز].align);

    إذا (mmq || k != kpad) {
        p = dq_configs[shader_سائيز].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_سائيز);
    }

    إذا (p == خالي_اشارو) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::سٽ_پڄاڻي;
        أرجع;
    }

    ثابت size_t x_sz = حجم(عائم) * x_ne;
    ثابت size_t y_sz = حجم(عائم) * y_ne;
    ثابت size_t qx_sz = x_ne * ggml_type_سائيز(quant)/ggml_blck_سائيز(quant);
    ثابت size_t qy_sz = mmq ? y_ne * ggml_type_سائيز(GGML_TYPE_Q8_1)/ggml_blck_سائيز(GGML_TYPE_Q8_1) : y_sz;
    ثابت size_t d_sz = حجم(عائم) * d_ne;
    عائم * x = (عائم *) يادگيري_ڏيو(x_sz);
    عائم * y = (عائم *) يادگيري_ڏيو(y_sz);
    فراغ * qx = يادگيري_ڏيو(qx_sz);
    vk_بفر qx_buf = ggml_vk_create_بفر_check(ctx->device, qx_sz, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    vk_بفر y_buf = ggml_vk_create_بفر_check(ctx->device, y_sz, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    vk_بفر qy_buf = ggml_vk_create_بفر_check(ctx->device, qy_sz, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    vk_بفر d_buf = ggml_vk_create_بفر_check(ctx->device, d_sz, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
    عائم * d = (عائم *) يادگيري_ڏيو(d_sz);
    عائم * d_chk = (عائم *) يادگيري_ڏيو(d_sz);

    لكل (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (عائم)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_معلومات(x, qx, x_ne, quant);

    لكل (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (عائم)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    إذا (ورهايو_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ورهايو_k_reduce, انگ_it);

        إذا (ctx->prealloc_ورهايو_k == خالي_اشارو || ctx->prealloc_ورهايو_k->size < حجم(عائم) * d_ne * ورهايو_k) {
            // Resize buffer
            إذا (ctx->prealloc_ورهايو_k != خالي_اشارو) {
                ggml_vk_destroy_بفر(ctx->prealloc_ورهايو_k);
            }
            ctx->prealloc_ورهايو_k = ggml_vk_create_بفر_check(ctx->device, حجم(عائم) * d_ne * ورهايو_k, {vk::يادگيري_Property_Flag_بٽ::eDeviceLocal});
        }
    }
    إذا (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, انگ_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_بفر_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_بفر_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    إذا (mmq) {
        لكل (size_t i = 0; i < انگ_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_ورهايو_k, 0, ctx->prealloc_سائيز_ورهايو_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                ورهايو_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_سائيز(GGML_TYPE_Q8_1)) * ggml_type_سائيز(GGML_TYPE_Q8_1), m * حجم(عائم)
            );
        }
    } وإلا {
        لكل (size_t i = 0; i < انگ_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_ورهايو_k, 0, ctx->prealloc_سائيز_ورهايو_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                ورهايو_k, batch, batch, batch, 1, 1, n,
                k * حجم(عائم), m * حجم(عائم)
            );
        }
    }
    ggml_vk_ctx_پڇاڙي(subctx);

    خودڪار begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, سچ, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_قطار_command_pools_cleanup(ctx->device);

    خودڪار end = std::chrono::high_resolution_clock::now();

    مزدوج وقت_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_بفر_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->معلومات = qx;
    src1_ggml->معلومات = y;
    tensor_ggml->معلومات = d_chk;

    ggml_cgraph * cgraph = ggml_new_گراف(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_گراف_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    مزدوج avg_err = 0.0;
    صحيح پهريون_err_n = -1;
    صحيح پهريون_err_m = -1;
    صحيح پهريون_err_b = -1;

    لكل (size_t i = 0; i < m*n*batch; i++) {
        مزدوج err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        إذا ((err > 0.05f || std::isnan(err)) && پهريون_err_n == -1) {
            پهريون_err_b = i / (m * n);
            پهريون_err_n = (i % (m * n)) / m;
            پهريون_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    مزدوج tflops = 2.0*m*n*k*batch*انگ_it / (وقت_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    إذا (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << ورهايو_k << " matmul " << وقت_ms / انگ_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::سٽ_پڄاڻي;

    إذا (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << پهريون_err_m << " n = " << پهريون_err_n << " b = " << پهريون_err_b << std::سٽ_پڄاڻي;
        std::cerr << "Actual result: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        ggml_vk_print_ميٽرڪس_area(d, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);
        std::cerr << std::سٽ_پڄاڻي;
        std::cerr << "Expected result: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        ggml_vk_print_ميٽرڪس_area(d_chk, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

        std::cerr << "src0: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        ggml_vk_print_ميٽرڪس_area(x, GGML_TYPE_F32, k, m, پهريون_err_m, پهريون_err_n, پهريون_err_b);
        std::cerr << std::سٽ_پڄاڻي;
        std::cerr << "src1: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
        ggml_vk_print_ميٽرڪس_area(y, GGML_TYPE_F32, k, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

        إذا (ورهايو_k > 1) {
            عائم * ورهايو_k_buf = (عائم *) يادگيري_ڏيو(حجم(عائم) * d_ne * ورهايو_k);
            ggml_vk_بفر_read(ctx->prealloc_ورهايو_k, 0, ورهايو_k_buf, حجم(عائم) * d_ne * ورهايو_k);

            std::cerr << "d_buf0: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            std::cerr << "d_buf1: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf + d_ne, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            std::cerr << "d_buf2: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            std::cerr << "d_buf3: " << std::سٽ_پڄاڻي << std::سٽ_پڄاڻي;
            ggml_vk_print_ميٽرڪس_area(ورهايو_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, پهريون_err_m, پهريون_err_n, پهريون_err_b);

            حرر(ورهايو_k_buf);
        }
    }

    ggml_vk_destroy_بفر(qx_buf);
    ggml_vk_destroy_بفر(y_buf);
    ggml_vk_destroy_بفر(qy_buf);
    ggml_vk_destroy_بفر(d_buf);

    حرر(x);
    حرر(qx);
    حرر(y);
    حرر(d);
    حرر(d_chk);
}
#جيڪڏهن_پورو

int64_t ggml_vk_get_op_batch_سائيز(ثابت ggml_tensor * op) {
    بدل (op->op) {
        حالة GGML_OP_GET_صفون:
            أرجع 0;
        حالة GGML_OP_MUL_MAT:
            أرجع op->ne[1];
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_پٺ:
            أرجع op->ne[2];
        افتراضي:
            أرجع ggml_nrows(op);
    }
}

#جيڪڏهن_وصف_ٿيل GGML_VULKAN_CHECK_RESULTS
ساكن فراغ ggml_vk_print_گراف_origin(ثابت ggml_tensor * tensor, std::vector<ثابت ggml_tensor *>& پورو, صحيح level = 0) {
    إذا (std::find(پورو.begin(), پورو.end(), tensor) != پورو.end() || level > 10) {
        أرجع;
    }
    لكل (صحيح j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != خالي_اشارو) << std::سٽ_پڄاڻي;

    پورو.push_back(tensor);

    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (tensor->src[i] != خالي_اشارو) {
            ggml_vk_print_گراف_origin(tensor->src[i], پورو, level + 1);
        }
    }
}

ساكن فراغ ggml_vk_print_tensor_area(ثابت ggml_tensor * tensor, ثابت فراغ * معلومات, صحيح i0, صحيح i1, صحيح i2, صحيح i3) {
    إذا (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        أرجع;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    لكل (صحيح idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    لكل (صحيح idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        لكل (صحيح idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            إذا (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                عائم val;
                إذا (tensor->type == GGML_TYPE_F32) {
                    val = *(ثابت عائم *) ((ثابت حرف *) معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } وإلا إذا (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_منزل_fp32(*(ثابت ggml_fp16_t *) ((ثابت حرف *) معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } وإلا إذا (tensor->type == GGML_TYPE_I32) {
                    val = *(ثابت int32_t *) ((ثابت حرف *) معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } وإلا {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } وإلا {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

ساكن فراغ ggml_vk_print_tensor(ثابت ggml_tensor * tensor, ثابت حرف * name) {
    فراغ * tensor_معلومات = tensor->معلومات;

    ثابت منطقي is_gpu = tensor->بفر != خالي_اشارو && ggml_backend_بفر_is_vk(tensor->بفر);

    إذا (is_gpu) {
        ثابت size_t tensor_سائيز = ggml_nbytes(tensor);
        tensor_معلومات = يادگيري_ڏيو(tensor_سائيز);

        ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)tensor->بفر->context;

        vk_بفر بفر_gpu = buf_ctx->dev_بفر;
        ggml_vk_بفر_read(بفر_gpu, vk_tensor_وٿي(tensor) + tensor->view_offs, tensor_معلومات, tensor_سائيز);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::سٽ_پڄاڻي;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::سٽ_پڄاڻي;
    إذا (tensor->src[0] != خالي_اشارو) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::سٽ_پڄاڻي;
    }
    إذا (tensor->src[1] != خالي_اشارو) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::سٽ_پڄاڻي;
    }
    std::cerr << std::سٽ_پڄاڻي << "Result:" << std::سٽ_پڄاڻي;
    ggml_vk_print_tensor_area(tensor, tensor_معلومات, 5, 5, 0, 0);
    std::cerr << std::سٽ_پڄاڻي;
    std::vector<ثابت ggml_tensor *> پورو;
    ggml_vk_print_گراف_origin(tensor, پورو);

    إذا (is_gpu) {
        حرر(tensor_معلومات);
    }
}

فراغ * comp_نتيجو;
size_t comp_سائيز;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
فراغ ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->انگ_additional_fused_ops];
    إذا (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_سيٽ_صفون) {
        أرجع;
    }

    check_counter++;
    إذا (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        أرجع;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    بنية ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ خطأ,
    };

    بنية ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<بنية ggml_tensor *, GGML_MAX_SRC> src_clone = {خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو};
    ثابت حرف * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<فراغ *> cloned_mallocs;

    بنية ggml_tensor * tensor_clone = خالي_اشارو;

    لكل (صحيح f = 0; f < ctx->انگ_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            إذا (srci == خالي_اشارو) {
                استمر;
            }
            // If a src tensor has been cloned, use that one
            خودڪار it = cloned_tensors.find(srci);
            إذا (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                استمر;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            size_t srci_سائيز = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            فراغ *src_بفر = يادگيري_ڏيو(srci_سائيز);
            cloned_mallocs.push_back(src_بفر);

            srci_clone->معلومات = src_بفر;
            إذا (ggml_backend_بفر_is_host(srci->بفر)) {
                يادگيري_نقل(srci_clone->معلومات, srci->معلومات, srci_سائيز);
                يادگيري_نقل(srci_clone->nb, srci->nb, حجم(size_t) * GGML_MAX_DIMS);
            } وإلا إذا (ggml_backend_بفر_is_vk(srci->بفر)) {
                ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)srci->بفر->context;
                vk_بفر& بفر_gpu = buf_ctx->dev_بفر;
                uint64_t وٿي = vk_tensor_وٿي(srci) + srci->view_offs;
                إذا (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    لكل (صحيح i3 = 0; i3 < srci->ne[3]; i3++) {
                        لكل (صحيح i2 = 0; i2 < srci->ne[2]; i2++) {
                            ثابت صحيح idx = i3*srci->ne[2] + i2;
                            ggml_vk_بفر_read(بفر_gpu, وٿي + idx * srci->nb[2], ((حرف *)srci_clone->معلومات + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    لكل (صحيح i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } وإلا {
                    إذا (وٿي + srci_سائيز >= بفر_gpu->size) {
                        srci_سائيز = بفر_gpu->size - وٿي;
                    }
                    ggml_vk_بفر_read(بفر_gpu, وٿي, srci_clone->معلومات, srci_سائيز);
                    يادگيري_نقل(srci_clone->nb, srci->nb, حجم(size_t) * GGML_MAX_DIMS);
                }
            } وإلا {
                GGML_ABORT("fatal error");
            }

            إذا (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        إذا (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ثابت عائم * params = (ثابت عائم *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            إذا (src_clone[4]) {
                ggml_flash_attn_ext_جوڙيو_sinks(tensor_clone, src_clone[4]);
            }
        } وإلا إذا (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } وإلا إذا (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(صحيح *)tensor->op_params);
        } وإلا إذا (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_گهڻائي) tensor->op_params[0]);
        } وإلا إذا (tensor->op == GGML_OP_SCALE) {
            ثابت عائم * params = (ثابت عائم *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } وإلا إذا (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_ARANGE) {
            ثابت عائم start = ggml_get_op_params_f32(tensor, 0);
            ثابت عائم stop = ggml_get_op_params_f32(tensor, 1);
            ثابت عائم قدم = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, قدم);
        } وإلا إذا (tensor->op == GGML_OP_FILL) {
            ثابت عائم value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } وإلا إذا (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } وإلا إذا (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_CLAMP) {
            ثابت عائم * params = (ثابت عائم *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } وإلا إذا (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } وإلا إذا (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } وإلا إذا (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } وإلا إذا (tensor->op == GGML_OP_REPEAT_پٺ) {
            tensor_clone = ggml_repeat_پٺ(ggml_ctx, src_clone[0], tensor);
        } وإلا إذا (tensor->op == GGML_OP_جوڙيو) {
            tensor_clone = ggml_جوڙيو(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } وإلا إذا (tensor->op == GGML_OP_سيٽ) {
            tensor_clone = ggml_سيٽ(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } وإلا إذا (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(عائم *)tensor->op_params);
        } وإلا إذا (tensor->op == GGML_OP_گروهه_NORM) {
            ثابت عائم * float_params = (ثابت عائم *)tensor->op_params;
            tensor_clone = ggml_گروهه_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } وإلا إذا (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(عائم *)tensor->op_params);
        } وإلا إذا (tensor->op == GGML_OP_RMS_NORM_پٺ) {
            ثابت عائم eps = ((عائم *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_پٺ(ggml_ctx, src_clone[0], src_clone[1], eps);
        } وإلا إذا (tensor->op == GGML_OP_SILU_پٺ) {
            tensor_clone = ggml_silu_پٺ(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_L2_NORM) {
            ثابت عائم eps = ((عائم *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } وإلا إذا (tensor->op == GGML_OP_SOFT_MAX) {
            إذا (tensor->src[1] != خالي_اشارو) {
                ثابت عائم * params = (ثابت عائم *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } وإلا {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } وإلا إذا (tensor->op == GGML_OP_SOFT_MAX_پٺ) {
            tensor_clone = ggml_soft_max_ext_پٺ(ggml_ctx, src_clone[0], src_clone[1], ((عائم *)tensor->op_params)[0], ((عائم *)tensor->op_params)[1]);
        } وإلا إذا (tensor->op == GGML_OP_DIAG_نقاب_INF) {
            tensor_clone = ggml_diag_نقاب_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } وإلا إذا (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_پٺ) {
            ثابت صحيح n_dims      = ((int32_t *) tensor->op_params)[1];
            ثابت صحيح mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            ثابت صحيح n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            ثابت عائم freq_بنياد       = ((عائم *) tensor->op_params)[5];
            ثابت عائم freq_scale      = ((عائم *) tensor->op_params)[6];
            ثابت عائم ext_ضرب_ڪندڙ      = ((عائم *) tensor->op_params)[7];
            ثابت عائم attn_ضرب_ڪندڙ     = ((عائم *) tensor->op_params)[8];
            ثابت عائم beta_fast       = ((عائم *) tensor->op_params)[9];
            ثابت عائم beta_slow       = ((عائم *) tensor->op_params)[10];
            إذا (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                إذا (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_بنياد, freq_scale, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow);
                } وإلا {
                    tensor_clone = ggml_rope_multi_پٺ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_بنياد, freq_scale, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow);
                }
            } وإلا {
                إذا (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_بنياد, freq_scale, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow);
                } وإلا {
                    tensor_clone = ggml_rope_ext_پٺ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_بنياد, freq_scale, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow);
                }
            }
            ثابت صحيح n_offs = ((int32_t *) tensor->op_params)[15];
            إذا (n_offs != 0) {
                tensor_clone = ggml_rope_سيٽ_وٿي(tensor_clone, n_offs);
            }
        } وإلا إذا (tensor->op == GGML_OP_UNARY) {
            بدل (ggml_get_unary_op(tensor)) {
            حالة GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_GELU_تيز:
                tensor_clone = ggml_gelu_تيز(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_سيٽ_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_سيٽ_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_سيٽ_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_سيٽ_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                اكسر;
            حالة GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_قدم:
                tensor_clone = ggml_قدم(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_ڦيرو:
                tensor_clone = ggml_ڦيرو(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                اكسر;
            حالة GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                اكسر;
            افتراضي:
                std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::سٽ_پڄاڻي;
                GGML_ABORT("fatal error");
            }
        } وإلا إذا (tensor->op == GGML_OP_GLU) {
            إذا (src_clone[1] == خالي_اشارو) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } وإلا {
                tensor_clone = ggml_glu_ورهايو(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_سيٽ_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_سيٽ_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } وإلا إذا (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            إذا (tensor->src[1] == خالي_اشارو) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } وإلا {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } وإلا إذا (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } وإلا إذا (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } وإلا إذا (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } وإلا إذا (tensor->op == GGML_OP_ترتيب_بدلايو) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_ترتيب_بدلايو(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } وإلا إذا (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_GET_صفون) {
            tensor_clone = ggml_get_صفون(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_ترتيب_ڏيو_ترتيب) *(صحيح *)tensor->op_params);
        } وإلا إذا (tensor->op == GGML_OP_چوٽي_K) {
            tensor_clone = ggml_چوٽي_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } وإلا إذا (tensor->op == GGML_OP_جوڙ) {
            tensor_clone = ggml_جوڙ(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_جوڙ_صفون) {
            tensor_clone = ggml_جوڙ_صفون(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } وإلا إذا (tensor->op == GGML_OP_DSV4_HC_PRE) {
            إذا (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } وإلا {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } وإلا إذا (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } وإلا إذا (tensor->op == GGML_OP_سراسري) {
            tensor_clone = ggml_سراسري(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } وإلا إذا (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_پٺ) {
            tensor_clone = ggml_cross_entropy_loss_پٺ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } وإلا إذا (tensor->op == GGML_OP_ڳڻپ_EQUAL) {
            tensor_clone = ggml_ڳڻپ_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], سچ, سچ, خطأ);
        } وإلا إذا (tensor->op == GGML_OP_IM2_کانو) {
            ثابت int32_t s0 = tensor->op_params[0];
            ثابت int32_t s1 = tensor->op_params[1];
            ثابت int32_t p0 = tensor->op_params[2];
            ثابت int32_t p1 = tensor->op_params[3];
            ثابت int32_t d0 = tensor->op_params[4];
            ثابت int32_t d1 = tensor->op_params[5];

            ثابت منطقي is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } وإلا إذا (tensor->op == GGML_OP_IM2_کانو_3_D) {
            ثابت int32_t s0 = tensor->op_params[0];
            ثابت int32_t s1 = tensor->op_params[1];
            ثابت int32_t s2 = tensor->op_params[2];
            ثابت int32_t p0 = tensor->op_params[3];
            ثابت int32_t p1 = tensor->op_params[4];
            ثابت int32_t p2 = tensor->op_params[5];
            ثابت int32_t d0 = tensor->op_params[6];
            ثابت int32_t d1 = tensor->op_params[7];
            ثابت int32_t d2 = tensor->op_params[8];
            ثابت int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } وإلا إذا (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            ثابت int32_t dim = tensor->op_params[0];
            ثابت int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } وإلا إذا (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            ثابت int32_t s0 = tensor->op_params[0];
            ثابت int32_t p0 = tensor->op_params[1];
            ثابت int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } وإلا إذا (tensor->op == GGML_OP_COL2IM_1D) {
            ثابت int32_t stride = tensor->op_params[0];
            ثابت int32_t oc     = tensor->op_params[1];
            ثابت int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } وإلا إذا (tensor->op == GGML_OP_POOL_1D) {
            تعداد ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ثابت int32_t k0 = tensor->op_params[1];
            ثابت int32_t s0 = tensor->op_params[2];
            ثابت int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } وإلا إذا (tensor->op == GGML_OP_POOL_2D) {
            تعداد ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            ثابت int32_t k0 = tensor->op_params[1];
            ثابت int32_t k1 = tensor->op_params[2];
            ثابت int32_t s0 = tensor->op_params[3];
            ثابت int32_t s1 = tensor->op_params[4];
            ثابت int32_t p0 = tensor->op_params[5];
            ثابت int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } وإلا إذا (tensor->op == GGML_OP_CONV_2D) {
            ثابت int32_t s0 = tensor->op_params[0];
            ثابت int32_t s1 = tensor->op_params[1];
            ثابت int32_t p0 = tensor->op_params[2];
            ثابت int32_t p1 = tensor->op_params[3];
            ثابت int32_t d0 = tensor->op_params[4];
            ثابت int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } وإلا إذا (tensor->op == GGML_OP_CONV_3D) {
            ثابت int32_t s0 = tensor->op_params[0];
            ثابت int32_t s1 = tensor->op_params[1];
            ثابت int32_t s2 = tensor->op_params[2];
            ثابت int32_t p0 = tensor->op_params[3];
            ثابت int32_t p1 = tensor->op_params[4];
            ثابت int32_t p2 = tensor->op_params[5];
            ثابت int32_t d0 = tensor->op_params[6];
            ثابت int32_t d1 = tensor->op_params[7];
            ثابت int32_t d2 = tensor->op_params[8];
            ثابت int32_t IC = tensor->op_params[9];
            ثابت int32_t N  = tensor->op_params[10];
            ثابت int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_سڌو(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } وإلا إذا (tensor->op == GGML_OP_CONV_2D_DW) {
            ثابت int32_t s0 = tensor->op_params[0];
            ثابت int32_t s1 = tensor->op_params[1];
            ثابت int32_t p0 = tensor->op_params[2];
            ثابت int32_t p1 = tensor->op_params[3];
            ثابت int32_t d0 = tensor->op_params[4];
            ثابت int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_سڌو(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } وإلا إذا (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            ثابت int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } وإلا إذا (tensor->op == GGML_OP_LEAKY_RELU) {
            ثابت عائم * op_params = (ثابت عائم *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], خطأ);
        } وإلا إذا (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } وإلا إذا (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } وإلا إذا (tensor->op == GGML_OP_GATED_لڪيري_ATTN) {
            ثابت عائم * op_params = (ثابت عائم *)tensor->op_params;
            tensor_clone = ggml_gated_لڪيري_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } وإلا إذا (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } وإلا إذا (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } وإلا إذا (tensor->op == GGML_OP_OPT_قدم_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_قدم_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } وإلا إذا (tensor->op == GGML_OP_OPT_قدم_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_قدم_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } وإلا إذا (tensor->op == GGML_OP_جوڙيو_ID) {
            tensor_clone = ggml_جوڙيو_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } وإلا إذا (tensor->op == GGML_OP_SSM_جاچ) {
            ثابت int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_جاچ(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } وإلا إذا (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } وإلا إذا (tensor->op == GGML_OP_ROLL) {
            ثابت int32_t s0 = tensor->op_params[0];
            ثابت int32_t s1 = tensor->op_params[1];
            ثابت int32_t s2 = tensor->op_params[2];
            ثابت int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        وإلا {
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::سٽ_پڄاڻي;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_گراف(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_گراف_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    إذا (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_سائيز = ggml_nbytes(tensor_clone);

    comp_نتيجو = يادگيري_ڏيو(comp_سائيز);
    يادگيري_نقل(comp_نتيجو, tensor_clone->معلومات, comp_سائيز);
    يادگيري_نقل(comp_nb, tensor_clone->nb, حجم(size_t) * GGML_MAX_DIMS);

    لكل (خودڪار m : cloned_mallocs) {
        حرر(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

فراغ ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, صحيح tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->انگ_additional_fused_ops];
    إذا (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_سيٽ_صفون) {
        أرجع;
    }

    إذا (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        أرجع;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    فراغ * tensor_معلومات = tensor->معلومات;

    إذا (ggml_backend_بفر_is_vk(tensor->بفر)) {
        size_t tensor_سائيز = ggml_nbytes(tensor);
        tensor_معلومات = يادگيري_ڏيو(tensor_سائيز);

        ggml_backend_vk_بفر_context * buf_ctx = (ggml_backend_vk_بفر_context *)tensor->بفر->context;

        vk_بفر& بفر_gpu = buf_ctx->dev_بفر;
        uint64_t وٿي = vk_tensor_وٿي(tensor) + tensor->view_offs;
        إذا (وٿي + tensor_سائيز >= بفر_gpu->size) {
            tensor_سائيز = بفر_gpu->size - وٿي;
        }

        ggml_vk_بفر_read(بفر_gpu, وٿي, tensor_معلومات, tensor_سائيز);
    }

    عائم پهريون_error_نتيجو = -1.0f;
    عائم پهريون_error_correct = -1.0f;
    std::array<صحيح, 4> پهريون_error = { -1, -1, -1, -1 };
    مزدوج avg_err = 0.0;
    size_t counter = 0;

    لكل (صحيح i3 = 0; i3 < tensor->ne[3]; i3++) {
        لكل (صحيح i2 = 0; i2 < tensor->ne[2]; i2++) {
            لكل (صحيح i1 = 0; i1 < tensor->ne[1]; i1++) {
                لكل (صحيح i0 = 0; i0 < tensor->ne[0]; i0++) {
                    ثابت منطقي بفر_سائيز_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_سائيز;
                    عائم correct = 0.0f;
                    عائم نتيجو = 0.0f;

                    إذا (بفر_سائيز_fit) {
                        إذا (tensor->type == GGML_TYPE_F32) {
                            correct = *(عائم *) ((حرف *) comp_نتيجو + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            نتيجو  = *(عائم *) ((حرف *) tensor_معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } وإلا إذا (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_منزل_fp32(*(ggml_fp16_t *) ((حرف *) comp_نتيجو + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            نتيجو  = ggml_fp16_منزل_fp32(*(ggml_fp16_t *) ((حرف *) tensor_معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } وإلا إذا (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_منزل_fp32(*(ggml_bf16_t *) ((حرف *) comp_نتيجو + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            نتيجو  = ggml_bf16_منزل_fp32(*(ggml_bf16_t *) ((حرف *) tensor_معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } وإلا إذا (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((حرف *) comp_نتيجو + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            نتيجو  = *(int32_t *) ((حرف *) tensor_معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } وإلا إذا (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((حرف *) comp_نتيجو + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            نتيجو  = *(int64_t *) ((حرف *) tensor_معلومات + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } وإلا {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::سٽ_پڄاڻي;
                        }
                    } وإلا {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::سٽ_پڄاڻي;
                        GGML_ABORT("fatal error");
                    }

                    إذا ((std::isnan(correct) != std::isnan(نتيجو)) || (std::isinf(correct) != std::isinf(نتيجو)) || !بفر_سائيز_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << نتيجو << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::سٽ_پڄاڻي;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::سٽ_پڄاڻي;
                        إذا (src0 != خالي_اشارو) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::سٽ_پڄاڻي;
                        }
                        إذا (src1 != خالي_اشارو) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::سٽ_پڄاڻي;
                        }
                        إذا (src2 != خالي_اشارو) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::سٽ_پڄاڻي;
                        }
                        إذا (src3 != خالي_اشارو) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::سٽ_پڄاڻي;
                        }
                        std::cerr << "First error: result=" << پهريون_error_نتيجو << " correct=" << پهريون_error_correct  << " i3=" << پهريون_error[3] << " i2=" << پهريون_error[2] << " i1=" << پهريون_error[1] << " i0=" << پهريون_error[0] << std::سٽ_پڄاڻي;
                        std::cerr << std::سٽ_پڄاڻي << "Result:" << std::سٽ_پڄاڻي;
                        ggml_vk_print_tensor_area(tensor, tensor_معلومات, i0, i1, i2, i3);
                        std::cerr << std::سٽ_پڄاڻي << "Correct:" << std::سٽ_پڄاڻي;
                        ggml_vk_print_tensor_area(tensor, comp_نتيجو, i0, i1, i2, i3);
                        std::cerr << std::سٽ_پڄاڻي;
                        std::vector<ثابت ggml_tensor *> پورو;
                        ggml_vk_print_گراف_origin(tensor, پورو);
                        GGML_ABORT("fatal error");
                    }
                    ثابت مزدوج denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    إذا (پهريون_error[0] == -1 && std::fabs(correct - نتيجو) / denom > 0.5) {
                        پهريون_error[0] = i0;
                        پهريون_error[1] = i1;
                        پهريون_error[2] = i2;
                        پهريون_error[3] = i3;
                        پهريون_error_نتيجو = نتيجو;
                        پهريون_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    إذا (!std::isinf(correct) && !std::isinf(نتيجو) && !std::isnan(correct) && !std::isnan(نتيجو)) {
                        avg_err += std::fabs(correct - نتيجو) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    إذا (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::سٽ_پڄاڻي;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::سٽ_پڄاڻي;
        إذا (src0 != خالي_اشارو) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::سٽ_پڄاڻي;
        }
        إذا (src1 != خالي_اشارو) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::سٽ_پڄاڻي;
        }
        إذا (src2 != خالي_اشارو) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::سٽ_پڄاڻي;
        }
        إذا (src3 != خالي_اشارو) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::سٽ_پڄاڻي;
        }
        std::cerr << "First error: result=" << پهريون_error_نتيجو << " correct=" << پهريون_error_correct  << " i3=" << پهريون_error[3] << " i2=" << پهريون_error[2] << " i1=" << پهريون_error[1] << " i0=" << پهريون_error[0] << std::سٽ_پڄاڻي;
        std::cerr << std::سٽ_پڄاڻي << "Result:" << std::سٽ_پڄاڻي;
        ggml_vk_print_tensor_area(tensor, tensor_معلومات, 5, 5, 0, 0);
        std::cerr << std::سٽ_پڄاڻي << "Correct:" << std::سٽ_پڄاڻي;
        ggml_vk_print_tensor_area(tensor, comp_نتيجو, 5, 5, 0, 0);
        std::cerr << std::سٽ_پڄاڻي;
        std::vector<ثابت ggml_tensor *> پورو;
        ggml_vk_print_گراف_origin(tensor, پورو);
    }

    إذا (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::سٽ_پڄاڻي;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::سٽ_پڄاڻي;
        إذا (src0 != خالي_اشارو) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::سٽ_پڄاڻي;
        }
        إذا (src1 != خالي_اشارو) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::سٽ_پڄاڻي;
        }
        إذا (src2 != خالي_اشارو) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::سٽ_پڄاڻي;
        }
        إذا (src3 != خالي_اشارو) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::سٽ_پڄاڻي;
        }
        std::cerr << "First error: result=" << پهريون_error_نتيجو << " correct=" << پهريون_error_correct  << " i3=" << پهريون_error[3] << " i2=" << پهريون_error[2] << " i1=" << پهريون_error[1] << " i0=" << پهريون_error[0] << std::سٽ_پڄاڻي;
        std::cerr << std::سٽ_پڄاڻي << "Result:" << std::سٽ_پڄاڻي;
        ggml_vk_print_tensor_area(tensor, tensor_معلومات, پهريون_error[0], پهريون_error[1], پهريون_error[2], پهريون_error[3]);
        std::cerr << std::سٽ_پڄاڻي << "Correct:" << std::سٽ_پڄاڻي;
        ggml_vk_print_tensor_area(tensor, comp_نتيجو, پهريون_error[0], پهريون_error[1], پهريون_error[2], پهريون_error[3]);
        std::cerr << std::سٽ_پڄاڻي;
        std::vector<ثابت ggml_tensor *> پورو;
        ggml_vk_print_گراف_origin(tensor, پورو);
        GGML_ABORT("fatal error");
    } وإلا {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::سٽ_پڄاڻي;
    }

    حرر(comp_نتيجو);
    comp_نتيجو = خالي_اشارو;
    comp_سائيز = 0;

    إذا (ggml_backend_بفر_is_vk(tensor->بفر)) {
        حرر(tensor_معلومات);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#جيڪڏهن_پورو

