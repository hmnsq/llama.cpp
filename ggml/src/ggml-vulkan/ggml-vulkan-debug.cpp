#समावेश "ggml-vulkan-common.h"

bool vk_memory_logger_enabled = false;

bool vk_perf_logger_enabled = false;

bool vk_perf_logger_concurrent = false;

bool vk_enable_sync_logger = false;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

व्योम vk_memory_logger::log_allocation(vk_बफर_ref buf_ref, माप_प्रकार size) {
    अगर (!vk_memory_logger_enabled) {
        वापस;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_बफर buf = buf_ref.lock();
    स्थिर bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    स्थिर std::string type = device ? "device" : "host";
    allocations[buf->बफर] = size;
    योग_device += device ? size : 0;
    योग_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_आकार(size) << " " << type << " at " << buf->बफर << ". Total device: " << format_आकार(योग_device) << ", total host: " << format_आकार(योग_host));
}

व्योम vk_memory_logger::log_deallocation(vk_बफर_ref buf_ref) {
    अगर (buf_ref.expired() || buf_ref.lock()->size == 0 || !vk_memory_logger_enabled) {
        वापस;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_बफर buf = buf_ref.lock();
    स्थिर bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::string type = device ? "device" : "host";
    स्वतः it = allocations.find(buf->बफर);
    अगर (it != allocations.end()) {
        योग_device -= device ? it->second : 0;
        योग_host -= device ? 0 : it->second;
        VK_LOG_MEMORY(buf->device->name << ": -" << format_आकार(it->second) << " " << type << " at " << buf->बफर << ". Total device: " << format_आकार(योग_device) << ", total host: " << format_आकार(योग_host));
        allocations.erase(it);
    } अन्यथा {
        VK_LOG_MEMORY("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->बफर);
    }
}

#अगर_घोषित GGML_VULKAN_CHECK_RESULTS
माप_प्रकार vk_skip_checks;
माप_प्रकार vk_output_tensor;

अटल व्योम ggml_vk_print_tensor(स्थिर ggml_tensor * tensor, स्थिर अक्षर * name);
#पूर्ण_अगर

#अगर_घोषित GGML_VULKAN_RUN_TESTS
अटल व्योम ggml_vk_print_matrix_area(स्थिर व्योम * तथ्य, ggml_type type, पूर्णांक ne0, पूर्णांक ne1, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2) {
    अगर (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        वापस;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    ख_लिखो(मानक_त्रुटि, "         ");
    क्रम (पूर्णांक idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        ख_लिखो(मानक_त्रुटि, "%7d ", idx1);
    }
    ख_लिखो(मानक_त्रुटि, "\n");
    क्रम (पूर्णांक idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        ख_लिखो(मानक_त्रुटि, "%7d: ", idx0);
        क्रम (पूर्णांक idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            अगर (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                भग्न val;
                अगर (type == GGML_TYPE_F32) {
                    val = *((स्थिर भग्न *) तथ्य + i2*ne1*ne0 + idx1*ne0 + idx0);
                } अन्यथा अगर (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*((स्थिर ggml_fp16_t *) तथ्य + i2*ne1*ne0 + idx1*ne0 + idx0));
                } अन्यथा {
                    GGML_ABORT("fatal error");
                }
                ख_लिखो(मानक_त्रुटि, "% 7.2f ", val);
            } अन्यथा {
                ख_लिखो(मानक_त्रुटि, "        ");
            }
        }
        ख_लिखो(मानक_त्रुटि, "\n");
    }
}

ढाँचा <typename X_TYPE, typename Y_TYPE>
व्योम ggml_vk_test_matmul(ggml_backend_vk_context * ctx, माप_प्रकार m, माप_प्रकार n, माप_प्रकार k, माप_प्रकार batch, माप_प्रकार अंक_it, पूर्णांक split_k, पूर्णांक shader_आकार) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << अंक_it << ", " << split_k << ", " << shader_आकार << ")");
    स्थिर माप_प्रकार x_ne = m * k * batch;
    स्थिर माप_प्रकार y_ne = k * n * batch;
    स्थिर माप_प्रकार d_ne = m * n * batch;

    ggml_type x_type = std::is_same<भग्न, X_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_type y_type = std::is_same<भग्न, Y_TYPE>() ? GGML_TYPE_F32 : GGML_TYPE_F16;
    vk_matmul_pipeline_कुंजी mm_test_कुंजी{x_type, y_type, false, false};
    स्वतः mm_test_it = ctx->device->pipeline_matmul.find(mm_test_कुंजी);
    GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.empty());
    स्वतः& mm_test_configs = mm_test_it->second;
    GGML_ASSERT(shader_आकार >= 0 && shader_आकार < (पूर्णांक)mm_test_configs.size());

    std::string shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_आकार);
    vk_pipeline p = mm_test_configs[shader_आकार].aligned ? mm_test_configs[shader_आकार].aligned : mm_test_configs[shader_आकार].unaligned;

    स्थिर माप_प्रकार kpad = ggml_vk_align_आकार(k, mm_test_configs[shader_आकार].align);

    अगर (k != kpad) {
        p = mm_test_configs[shader_आकार].unaligned;
        shname = std::string(ggml_type_name(x_type)) + "_" + std::string(ggml_type_name(y_type)) + "_" + std::to_string(shader_आकार);
    }

    अगर (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, अंक_it);

        अगर (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < माप(भग्न) * d_ne * split_k) {
            // Resize buffer
            अगर (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_बफर(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_बफर_check(ctx->device, माप(भग्न) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_बफर d_X = ggml_vk_create_बफर_check(ctx->device, माप(X_TYPE) * x_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_बफर d_Y = ggml_vk_create_बफर_check(ctx->device, माप(Y_TYPE) * y_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_बफर d_D = ggml_vk_create_बफर_check(ctx->device, माप(भग्न) * d_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) दो_स्मृति(माप(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) दो_स्मृति(माप(Y_TYPE) * y_ne);
    भग्न* d = (भग्न *) दो_स्मृति(माप(भग्न) * d_ne);

    क्रम (माप_प्रकार i = 0; i < x_ne; i++) {
        अगर (std::is_same<भग्न, X_TYPE>()) {
            x[i] = (अक्रम() / (भग्न)अक्रम_उच्च) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } अन्यथा अगर (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_to_fp16((अक्रम() / (भग्न)अक्रम_उच्च) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } अन्यथा {
            GGML_ABORT("fatal error");
        }
    }
    क्रम (माप_प्रकार i = 0; i < y_ne; i++) {
        अगर (std::is_same<भग्न, Y_TYPE>()) {
            y[i] = (अक्रम() / (भग्न)अक्रम_उच्च) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } अन्यथा अगर (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_to_fp16((अक्रम() / (भग्न)अक्रम_उच्च) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } अन्यथा {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_बफर_write(d_X, 0, x, माप(X_TYPE) * k * m * batch);
    ggml_vk_बफर_write(d_Y, 0, y, माप(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    क्रम (माप_प्रकार i = 0; i < अंक_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(ctx, d_X), ggml_vk_subbuffer(ctx, d_Y), ggml_vk_subbuffer(ctx, d_D), ggml_vk_subbuffer(ctx, ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1, n,
            k * माप(Y_TYPE), m * माप(भग्न)
        );
    }
    ggml_vk_ctx_end(subctx);

    स्वतः begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    स्वतः end = std::chrono::high_resolution_clock::now();
    द्विगुन समय = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_बफर_read(d_D, 0, d, माप(भग्न) * d_ne);

    भग्न * d_chk = (भग्न *) दो_स्मृति(माप(भग्न) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    अगर (std::is_same<भग्न, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } अन्यथा अगर (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }
    अगर (std::is_same<भग्न, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } अन्यथा अगर (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } अन्यथा {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->तथ्य = x;
    src1_ggml->तथ्य = y;
    tensor_ggml->तथ्य = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    द्विगुन avg_err = 0.0;
    पूर्णांक first_err_n = -1;
    पूर्णांक first_err_m = -1;
    पूर्णांक first_err_b = -1;

    क्रम (माप_प्रकार i = 0; i < m*n*batch; i++) {
        द्विगुन err = std::भ_असल(d[i] - d_chk[i]);
        avg_err += err;

        अगर ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    द्विगुन tflops = 2.0*m*n*k*batch*अंक_it / (समय / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::अ_त्रुटि << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << समय / अंक_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::प_पूर्ण;

    अगर (avg_err > 0.1 || std::isnan(avg_err)) {
        std::अ_त्रुटि << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::प_पूर्ण;
        std::अ_त्रुटि << "Actual result: " << std::प_पूर्ण << std::प_पूर्ण;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::अ_त्रुटि << "Expected result: " << std::प_पूर्ण << std::प_पूर्ण;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        अगर (split_k > 1) {
            भग्न * split_k_buf = (भग्न *) दो_स्मृति(माप(भग्न) * d_ne * split_k);
            ggml_vk_बफर_read(ctx->prealloc_split_k, 0, split_k_buf, माप(भग्न) * d_ne * split_k);

            std::अ_त्रुटि << "d_buf0: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::अ_त्रुटि << "d_buf1: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::अ_त्रुटि << "d_buf2: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::अ_त्रुटि << "d_buf3: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            मुक्त(split_k_buf);
        }
    }

    मुक्त(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    ggml_vk_destroy_बफर(d_X);
    ggml_vk_destroy_बफर(d_Y);
    ggml_vk_destroy_बफर(d_D);

    मुक्त(x);
    मुक्त(y);
    मुक्त(d);
}

ढाँचा व्योम ggml_vk_test_matmul<ggml_fp16_t, भग्न>(ggml_backend_vk_context * ctx, माप_प्रकार m, माप_प्रकार n, माप_प्रकार k, माप_प्रकार batch, माप_प्रकार अंक_it, पूर्णांक split_k, पूर्णांक shader_आकार);

अटल व्योम ggml_vk_print_tensor_area(स्थिर ggml_tensor * tensor, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3) {
    अगर (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        वापस;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    ख_लिखो(मानक_त्रुटि, "         ");
    क्रम (पूर्णांक idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        ख_लिखो(मानक_त्रुटि, "%7d ", idx1);
    }
    ख_लिखो(मानक_त्रुटि, "\n");
    क्रम (पूर्णांक idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        ख_लिखो(मानक_त्रुटि, "%7d: ", idx0);
        क्रम (पूर्णांक idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            अगर (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                भग्न val;
                अगर (tensor->type == GGML_TYPE_F32) {
                    val = *(भग्न *) ((अक्षर *) tensor->तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } अन्यथा अगर (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((अक्षर *) tensor->तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } अन्यथा {
                    GGML_ABORT("fatal error");
                }
                ख_लिखो(मानक_त्रुटि, "% 7.2f ", val);
            } अन्यथा {
                ख_लिखो(मानक_त्रुटि, "        ");
            }
        }
        ख_लिखो(मानक_त्रुटि, "\n");
    }
}

अटल व्योम ggml_vk_quantize_तथ्य(स्थिर भग्न * from, व्योम * to, माप_प्रकार ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, to, 0, 1, ne, nullptr);
}

अटल व्योम ggml_vk_dequantize_तथ्य(स्थिर व्योम * from, भग्न * to, माप_प्रकार ne, ggml_type quant) {
    अगर (quant == GGML_TYPE_F32) {
        स_नकल(to, from, माप(भग्न) * ne);
        वापस;
    }

    स्थिर स्वतः * tt = ggml_get_type_traits(quant);

    ggml_to_float_t dequant_fn = tt->to_float;

    dequant_fn(from, to, ne);
}

व्योम ggml_vk_test_dequant(ggml_backend_vk_context * ctx, माप_प्रकार ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    स्थिर माप_प्रकार x_sz = माप(भग्न) * ne;
    स्थिर माप_प्रकार x_sz_f16 = माप(ggml_fp16_t) * ne;
    स्थिर माप_प्रकार qx_sz = ne * ggml_type_आकार(quant)/ggml_blck_आकार(quant);
    भग्न * x = (भग्न *) दो_स्मृति(x_sz);
    व्योम * qx = दो_स्मृति(qx_sz);
    vk_बफर qx_buf = ggml_vk_create_बफर_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_बफर x_buf = ggml_vk_create_बफर_check(ctx->device, x_sz_f16, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    भग्न * x_ref = (भग्न *) दो_स्मृति(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) दो_स्मृति(x_sz_f16);

    क्रम (माप_प्रकार i = 0; i < ne; i++) {
        x[i] = अक्रम() / (भग्न)अक्रम_उच्च;
    }

    vk_pipeline p = ggml_vk_get_to_fp16(ctx, quant);

    ggml_vk_quantize_तथ्य(x, qx, ne, quant);
    ggml_vk_dequantize_तथ्य(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_बफर_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    स्थिर std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_end(subctx);

    स्वतः begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    स्वतः end = std::chrono::high_resolution_clock::now();

    द्विगुन ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_बफर_read(x_buf, 0, x_chk, x_sz_f16);

    पूर्णांक first_err = -1;

    द्विगुन avg_err = 0.0;
    क्रम (माप_प्रकार i = 0; i < ne; i++) {
        द्विगुन error = std::भ_असल(x_ref[i] - ggml_fp16_to_fp32(x_chk[i]));
        avg_err += error;

        अगर (first_err < 0 && error > 0.05) {
            first_err = i;
        }
    }

    avg_err /= ne;

    std::अ_त्रुटि << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::प_पूर्ण;

    अगर (avg_err > 0.1) {
        std::अ_त्रुटि << "first_error = " << first_err << std::प_पूर्ण;
        std::अ_त्रुटि << "Actual result: " << std::प_पूर्ण << std::प_पूर्ण;
        क्रम (पूर्णांक i = std::max(0, first_err - 5); i < std::min((पूर्णांक)ne, first_err + 5); i++) {
            std::अ_त्रुटि << ggml_fp16_to_fp32(x_chk[i]) << ", ";
        }
        std::अ_त्रुटि << std::प_पूर्ण << "Expected result: " << std::प_पूर्ण << std::प_पूर्ण;
        क्रम (पूर्णांक i = std::max(0, first_err - 5); i < std::min((पूर्णांक)ne, first_err + 5); i++) {
            std::अ_त्रुटि << x_ref[i] << ", ";
        }
        std::अ_त्रुटि << std::प_पूर्ण;
    }

    ggml_vk_destroy_बफर(x_buf);
    ggml_vk_destroy_बफर(qx_buf);

    मुक्त(x);
    मुक्त(qx);
    मुक्त(x_ref);
    मुक्त(x_chk);
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

व्योम ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, माप_प्रकार m, माप_प्रकार n, माप_प्रकार k, माप_प्रकार batch, माप_प्रकार अंक_it, माप_प्रकार split_k, माप_प्रकार shader_आकार, ggml_type quant, bool mmq) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << अंक_it << ", " << split_k << ", " << ggml_type_name(quant) << ")");
    स्थिर माप_प्रकार x_ne = m * k * batch;
    स्थिर माप_प्रकार y_ne = k * n * batch;
    स्थिर माप_प्रकार d_ne = m * n * batch;

    ggml_type b_type = mmq ? GGML_TYPE_Q8_1 : GGML_TYPE_F32;
    bool f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_कुंजी dq_कुंजी{quant, b_type, false, f16acc};
    स्वतः dq_it = ctx->device->pipeline_matmul.find(dq_कुंजी);
    अगर (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        अगर (f16acc) {
            dq_कुंजी.f16acc = false;
            dq_it = ctx->device->pipeline_matmul.find(dq_कुंजी);
        }
    }
    अगर (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        std::अ_त्रुटि << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::प_पूर्ण;
        वापस;
    }
    स्वतः& dq_configs = dq_it->second;
    अगर (shader_आकार >= (पूर्णांक)dq_configs.size()) {
        std::अ_त्रुटि << "error: shader_size " << shader_आकार << " >= configs.size() " << dq_configs.size() << " for " << ggml_type_name(quant) << std::प_पूर्ण;
        वापस;
    }

    std::string shname = std::string(ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_आकार);
    vk_pipeline p = dq_configs[shader_आकार].aligned ? dq_configs[shader_आकार].aligned : dq_configs[shader_आकार].unaligned;

    स्थिर माप_प्रकार kpad = mmq ? 0 : ggml_vk_align_आकार(k, dq_configs[shader_आकार].align);

    अगर (mmq || k != kpad) {
        p = dq_configs[shader_आकार].unaligned;
        shname = std::string(ggml_type_name(quant)) + "_" + std::to_string(shader_आकार);
    }

    अगर (p == nullptr) {
        std::अ_त्रुटि << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::प_पूर्ण;
        वापस;
    }

    स्थिर माप_प्रकार x_sz = माप(भग्न) * x_ne;
    स्थिर माप_प्रकार y_sz = माप(भग्न) * y_ne;
    स्थिर माप_प्रकार qx_sz = x_ne * ggml_type_आकार(quant)/ggml_blck_आकार(quant);
    स्थिर माप_प्रकार qy_sz = mmq ? y_ne * ggml_type_आकार(GGML_TYPE_Q8_1)/ggml_blck_आकार(GGML_TYPE_Q8_1) : y_sz;
    स्थिर माप_प्रकार d_sz = माप(भग्न) * d_ne;
    भग्न * x = (भग्न *) दो_स्मृति(x_sz);
    भग्न * y = (भग्न *) दो_स्मृति(y_sz);
    व्योम * qx = दो_स्मृति(qx_sz);
    vk_बफर qx_buf = ggml_vk_create_बफर_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_बफर y_buf = ggml_vk_create_बफर_check(ctx->device, y_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_बफर qy_buf = ggml_vk_create_बफर_check(ctx->device, qy_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_बफर d_buf = ggml_vk_create_बफर_check(ctx->device, d_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    भग्न * d = (भग्न *) दो_स्मृति(d_sz);
    भग्न * d_chk = (भग्न *) दो_स्मृति(d_sz);

    क्रम (माप_प्रकार i = 0; i < x_ne; i++) {
        x[i] = (अक्रम() / (भग्न)अक्रम_उच्च) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_तथ्य(x, qx, x_ne, quant);

    क्रम (माप_प्रकार i = 0; i < y_ne; i++) {
        y[i] = (अक्रम() / (भग्न)अक्रम_उच्च) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    अगर (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, अंक_it);

        अगर (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < माप(भग्न) * d_ne * split_k) {
            // Resize buffer
            अगर (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_बफर(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_बफर_check(ctx->device, माप(भग्न) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }
    अगर (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, अंक_it);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_बफर_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_बफर_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    अगर (mmq) {
        क्रम (माप_प्रकार i = 0; i < अंक_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_आकार_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / ggml_blck_आकार(GGML_TYPE_Q8_1)) * ggml_type_आकार(GGML_TYPE_Q8_1), m * माप(भग्न)
            );
        }
    } अन्यथा {
        क्रम (माप_प्रकार i = 0; i < अंक_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_आकार_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                k * माप(भग्न), m * माप(भग्न)
            );
        }
    }
    ggml_vk_ctx_end(subctx);

    स्वतः begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    स्वतः end = std::chrono::high_resolution_clock::now();

    द्विगुन time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_बफर_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->तथ्य = qx;
    src1_ggml->तथ्य = y;
    tensor_ggml->तथ्य = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    द्विगुन avg_err = 0.0;
    पूर्णांक first_err_n = -1;
    पूर्णांक first_err_m = -1;
    पूर्णांक first_err_b = -1;

    क्रम (माप_प्रकार i = 0; i < m*n*batch; i++) {
        द्विगुन err = std::भ_असल(d[i] - d_chk[i]);
        avg_err += err;

        अगर ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    द्विगुन tflops = 2.0*m*n*k*batch*अंक_it / (time_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::अ_त्रुटि << "TEST dequant matmul " << shname;
    अगर (mmq) {
        std::अ_त्रुटि << " mmq";
    }
    std::अ_त्रुटि << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time_ms / अंक_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::प_पूर्ण;

    अगर (avg_err > 0.01 || std::isnan(avg_err)) {
        std::अ_त्रुटि << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::प_पूर्ण;
        std::अ_त्रुटि << "Actual result: " << std::प_पूर्ण << std::प_पूर्ण;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::अ_त्रुटि << std::प_पूर्ण;
        std::अ_त्रुटि << "Expected result: " << std::प_पूर्ण << std::प_पूर्ण;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        std::अ_त्रुटि << "src0: " << std::प_पूर्ण << std::प_पूर्ण;
        ggml_vk_print_matrix_area(x, GGML_TYPE_F32, k, m, first_err_m, first_err_n, first_err_b);
        std::अ_त्रुटि << std::प_पूर्ण;
        std::अ_त्रुटि << "src1: " << std::प_पूर्ण << std::प_पूर्ण;
        ggml_vk_print_matrix_area(y, GGML_TYPE_F32, k, n, first_err_m, first_err_n, first_err_b);

        अगर (split_k > 1) {
            भग्न * split_k_buf = (भग्न *) दो_स्मृति(माप(भग्न) * d_ne * split_k);
            ggml_vk_बफर_read(ctx->prealloc_split_k, 0, split_k_buf, माप(भग्न) * d_ne * split_k);

            std::अ_त्रुटि << "d_buf0: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::अ_त्रुटि << "d_buf1: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::अ_त्रुटि << "d_buf2: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::अ_त्रुटि << "d_buf3: " << std::प_पूर्ण << std::प_पूर्ण;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            मुक्त(split_k_buf);
        }
    }

    ggml_vk_destroy_बफर(qx_buf);
    ggml_vk_destroy_बफर(y_buf);
    ggml_vk_destroy_बफर(qy_buf);
    ggml_vk_destroy_बफर(d_buf);

    मुक्त(x);
    मुक्त(qx);
    मुक्त(y);
    मुक्त(d);
    मुक्त(d_chk);
}
#पूर्ण_अगर

int64_t ggml_vk_get_op_batch_आकार(स्थिर ggml_tensor * op) {
    चयन (op->op) {
        हाल GGML_OP_GET_ROWS:
            वापस 0;
        हाल GGML_OP_MUL_MAT:
            वापस op->ne[1];
        हाल GGML_OP_MUL_MAT_ID:
        हाल GGML_OP_ROPE:
        हाल GGML_OP_ROPE_BACK:
            वापस op->ne[2];
        शेष:
            वापस ggml_nrows(op);
    }
}

#अगर_घोषित GGML_VULKAN_CHECK_RESULTS
अटल व्योम ggml_vk_print_graph_origin(स्थिर ggml_tensor * tensor, std::vector<स्थिर ggml_tensor *>& done, पूर्णांक level = 0) {
    अगर (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) {
        वापस;
    }
    क्रम (पूर्णांक j = 0; j < level; j++) {
        std::अ_त्रुटि << " ";
    }
    std::अ_त्रुटि << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::प_पूर्ण;

    done.push_back(tensor);

    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (tensor->src[i] != nullptr) {
            ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

अटल व्योम ggml_vk_print_tensor_area(स्थिर ggml_tensor * tensor, स्थिर व्योम * तथ्य, पूर्णांक i0, पूर्णांक i1, पूर्णांक i2, पूर्णांक i3) {
    अगर (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        वापस;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    ख_लिखो(मानक_त्रुटि, "         ");
    क्रम (पूर्णांक idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        ख_लिखो(मानक_त्रुटि, "%7d ", idx1);
    }
    ख_लिखो(मानक_त्रुटि, "\n");
    क्रम (पूर्णांक idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        ख_लिखो(मानक_त्रुटि, "%7d: ", idx0);
        क्रम (पूर्णांक idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            अगर (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                भग्न val;
                अगर (tensor->type == GGML_TYPE_F32) {
                    val = *(स्थिर भग्न *) ((स्थिर अक्षर *) तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } अन्यथा अगर (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(स्थिर ggml_fp16_t *) ((स्थिर अक्षर *) तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } अन्यथा अगर (tensor->type == GGML_TYPE_I32) {
                    val = *(स्थिर int32_t *) ((स्थिर अक्षर *) तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } अन्यथा {
                    GGML_ABORT("fatal error");
                }
                ख_लिखो(मानक_त्रुटि, "% 7.2f ", val);
            } अन्यथा {
                ख_लिखो(मानक_त्रुटि, "        ");
            }
        }
        ख_लिखो(मानक_त्रुटि, "\n");
    }
}

अटल व्योम ggml_vk_print_tensor(स्थिर ggml_tensor * tensor, स्थिर अक्षर * name) {
    व्योम * tensor_तथ्य = tensor->तथ्य;

    स्थिर bool is_gpu = tensor->बफर != nullptr && ggml_backend_बफर_is_vk(tensor->बफर);

    अगर (is_gpu) {
        स्थिर माप_प्रकार tensor_आकार = ggml_nbytes(tensor);
        tensor_तथ्य = दो_स्मृति(tensor_आकार);

        ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;

        vk_बफर बफर_gpu = buf_ctx->dev_बफर;
        ggml_vk_बफर_read(बफर_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_तथ्य, tensor_आकार);
    }

    std::अ_त्रुटि << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::प_पूर्ण;
    std::अ_त्रुटि << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::प_पूर्ण;
    अगर (tensor->src[0] != nullptr) {
        std::अ_त्रुटि << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::प_पूर्ण;
    }
    अगर (tensor->src[1] != nullptr) {
        std::अ_त्रुटि << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::प_पूर्ण;
    }
    std::अ_त्रुटि << std::प_पूर्ण << "Result:" << std::प_पूर्ण;
    ggml_vk_print_tensor_area(tensor, tensor_तथ्य, 5, 5, 0, 0);
    std::अ_त्रुटि << std::प_पूर्ण;
    std::vector<स्थिर ggml_tensor *> done;
    ggml_vk_print_graph_origin(tensor, done);

    अगर (is_gpu) {
        मुक्त(tensor_तथ्य);
    }
}

व्योम * comp_परिणाम;
माप_प्रकार comp_आकार;
माप_प्रकार comp_nb[GGML_MAX_DIMS];
माप_प्रकार check_गणक = 0;
व्योम ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णांक tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->अंक_additional_fused_ops];
    अगर (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        वापस;
    }

    check_गणक++;
    अगर (!(vk_output_tensor > 0 && vk_output_tensor == check_गणक) && check_गणक <= vk_skip_checks) {
        वापस;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    काष्ठा ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ false,
    };

    काष्ठा ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<काष्ठा ggml_tensor *, GGML_MAX_SRC> src_clone = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    स्थिर अक्षर * srci_name[GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<ggml_tensor *, ggml_tensor *> cloned_tensors;
    std::vector<व्योम *> cloned_mallocs;

    काष्ठा ggml_tensor * tensor_clone = nullptr;

    क्रम (पूर्णांक f = 0; f < ctx->अंक_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * srci = tensor->src[i];
            अगर (srci == nullptr) {
                जारी;
            }
            // If a src tensor has been cloned, use that one
            स्वतः it = cloned_tensors.find(srci);
            अगर (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                जारी;
            }
            ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
            माप_प्रकार srci_आकार = ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            व्योम *src_बफर = दो_स्मृति(srci_आकार);
            cloned_mallocs.push_back(src_बफर);

            srci_clone->तथ्य = src_बफर;
            अगर (ggml_backend_बफर_is_host(srci->बफर)) {
                स_नकल(srci_clone->तथ्य, srci->तथ्य, srci_आकार);
                स_नकल(srci_clone->nb, srci->nb, माप(माप_प्रकार) * GGML_MAX_DIMS);
            } अन्यथा अगर (ggml_backend_बफर_is_vk(srci->बफर)) {
                ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)srci->बफर->context;
                vk_बफर& बफर_gpu = buf_ctx->dev_बफर;
                uint64_t offset = vk_tensor_offset(srci) + srci->view_offs;
                अगर (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                    क्रम (पूर्णांक i3 = 0; i3 < srci->ne[3]; i3++) {
                        क्रम (पूर्णांक i2 = 0; i2 < srci->ne[2]; i2++) {
                            स्थिर पूर्णांक idx = i3*srci->ne[2] + i2;
                            ggml_vk_बफर_read(बफर_gpu, offset + idx * srci->nb[2], ((अक्षर *)srci_clone->तथ्य + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    क्रम (पूर्णांक i = 2; i < GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } अन्यथा {
                    अगर (offset + srci_आकार >= बफर_gpu->size) {
                        srci_आकार = बफर_gpu->size - offset;
                    }
                    ggml_vk_बफर_read(बफर_gpu, offset, srci_clone->तथ्य, srci_आकार);
                    स_नकल(srci_clone->nb, srci->nb, माप(माप_प्रकार) * GGML_MAX_DIMS);
                }
            } अन्यथा {
                GGML_ABORT("fatal error");
            }

            अगर (vk_output_tensor > 0 && vk_output_tensor == check_गणक) {
                ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        अगर (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            स्थिर भग्न * params = (स्थिर भग्न *)tensor->op_params;
            tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            अगर (src_clone[4]) {
                ggml_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]);
            }
        } अन्यथा अगर (tensor->op == GGML_OP_MUL_MAT) {
            tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_MUL_MAT_ID) {
            tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } अन्यथा अगर (tensor->op == GGML_OP_SUB) {
            tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_MUL) {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_DIV) {
            tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_CONCAT) {
            tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(पूर्णांक *)tensor->op_params);
        } अन्यथा अगर (tensor->op == GGML_OP_UPSCALE) {
            tensor_clone = ggml_interpolate(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_mode) tensor->op_params[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_SCALE) {
            स्थिर भग्न * params = (स्थिर भग्न *)tensor->op_params;
            tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_ADD1) {
            tensor_clone = ggml_add1(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_ARANGE) {
            स्थिर भग्न start = ggml_get_op_params_f32(tensor, 0);
            स्थिर भग्न stop = ggml_get_op_params_f32(tensor, 1);
            स्थिर भग्न step = ggml_get_op_params_f32(tensor, 2);
            tensor_clone = ggml_arange(ggml_ctx, start, stop, step);
        } अन्यथा अगर (tensor->op == GGML_OP_FILL) {
            स्थिर भग्न value = ggml_get_op_params_f32(tensor, 0);
            tensor_clone = ggml_fill(ggml_ctx, src_clone[0], value);
        } अन्यथा अगर (tensor->op == GGML_OP_SQR) {
            tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_SQRT) {
            tensor_clone = ggml_sqrt(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_SIN) {
            tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_COS) {
            tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_LOG) {
            tensor_clone = ggml_log(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_TRI) {
            tensor_clone = ggml_tri(ggml_ctx, src_clone[0], (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
        } अन्यथा अगर (tensor->op == GGML_OP_DIAG) {
            tensor_clone = ggml_diag(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_CLAMP) {
            स्थिर भग्न * params = (स्थिर भग्न *)tensor->op_params;
            tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_PAD) {
            tensor_clone = ggml_pad_ext(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } अन्यथा अगर (tensor->op == GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = ggml_pad_reflect_1d(ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_REPEAT) {
            tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
        } अन्यथा अगर (tensor->op == GGML_OP_REPEAT_BACK) {
            tensor_clone = ggml_repeat_back(ggml_ctx, src_clone[0], tensor);
        } अन्यथा अगर (tensor->op == GGML_OP_ADD) {
            tensor_clone = ggml_add(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_ACC) {
            tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } अन्यथा अगर (tensor->op == GGML_OP_SET) {
            tensor_clone = ggml_set(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } अन्यथा अगर (tensor->op == GGML_OP_NORM) {
            tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(भग्न *)tensor->op_params);
        } अन्यथा अगर (tensor->op == GGML_OP_GROUP_NORM) {
            स्थिर भग्न * float_params = (स्थिर भग्न *)tensor->op_params;
            tensor_clone = ggml_group_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_RMS_NORM) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(भग्न *)tensor->op_params);
        } अन्यथा अगर (tensor->op == GGML_OP_RMS_NORM_BACK) {
            स्थिर भग्न eps = ((भग्न *) tensor->op_params)[0];
            tensor_clone = ggml_rms_norm_back(ggml_ctx, src_clone[0], src_clone[1], eps);
        } अन्यथा अगर (tensor->op == GGML_OP_SILU_BACK) {
            tensor_clone = ggml_silu_back(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_L2_NORM) {
            स्थिर भग्न eps = ((भग्न *) tensor->op_params)[0];
            tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
        } अन्यथा अगर (tensor->op == GGML_OP_SOFT_MAX) {
            अगर (tensor->src[1] != nullptr) {
                स्थिर भग्न * params = (स्थिर भग्न *)tensor->op_params;
                tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } अन्यथा {
                tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
            }
        } अन्यथा अगर (tensor->op == GGML_OP_SOFT_MAX_BACK) {
            tensor_clone = ggml_soft_max_ext_back(ggml_ctx, src_clone[0], src_clone[1], ((भग्न *)tensor->op_params)[0], ((भग्न *)tensor->op_params)[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_DIAG_MASK_INF) {
            tensor_clone = ggml_diag_mask_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_BACK) {
            स्थिर पूर्णांक n_dims      = ((int32_t *) tensor->op_params)[1];
            स्थिर पूर्णांक mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            स्थिर पूर्णांक n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            स्थिर भग्न freq_base       = ((भग्न *) tensor->op_params)[5];
            स्थिर भग्न freq_scale      = ((भग्न *) tensor->op_params)[6];
            स्थिर भग्न ext_factor      = ((भग्न *) tensor->op_params)[7];
            स्थिर भग्न attn_factor     = ((भग्न *) tensor->op_params)[8];
            स्थिर भग्न beta_fast       = ((भग्न *) tensor->op_params)[9];
            स्थिर भग्न beta_slow       = ((भग्न *) tensor->op_params)[10];
            अगर (mode & GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                अगर (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } अन्यथा {
                    tensor_clone = ggml_rope_multi_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            } अन्यथा {
                अगर (tensor->op == GGML_OP_ROPE) {
                    tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } अन्यथा {
                    tensor_clone = ggml_rope_ext_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            }
            स्थिर पूर्णांक n_offs = ((int32_t *) tensor->op_params)[15];
            अगर (n_offs != 0) {
                tensor_clone = ggml_rope_set_offset(tensor_clone, n_offs);
            }
        } अन्यथा अगर (tensor->op == GGML_OP_UNARY) {
            चयन (ggml_get_unary_op(tensor)) {
            हाल GGML_UNARY_OP_EXP:
                tensor_clone = ggml_exp(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_EXPM1:
                tensor_clone = ggml_expm1(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_ELU:
                tensor_clone = ggml_elu(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_SILU:
                tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_GELU:
                tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_GELU_ERF:
                tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_GELU_QUICK:
                tensor_clone = ggml_gelu_quick(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_RELU:
                tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_XIELU:
                tensor_clone = ggml_xielu(ggml_ctx, src_clone[0], 0, 0, 0, 0);
                ggml_set_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
                ggml_set_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
                ggml_set_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
                ggml_set_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
                अवरोध;
            हाल GGML_UNARY_OP_NEG:
                tensor_clone = ggml_neg(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_TANH:
                tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_SIGMOID:
                tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = ggml_hardsigmoid(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_HARDSWISH:
                tensor_clone = ggml_hardswish(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_ABS:
                tensor_clone = ggml_abs(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = ggml_softplus(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_STEP:
                tensor_clone = ggml_step(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_ROUND:
                tensor_clone = ggml_round(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_CEIL:
                tensor_clone = ggml_ceil(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_FLOOR:
                tensor_clone = ggml_floor(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_TRUNC:
                tensor_clone = ggml_trunc(ggml_ctx, src_clone[0]);
                अवरोध;
            हाल GGML_UNARY_OP_SGN:
                tensor_clone = ggml_sgn(ggml_ctx, src_clone[0]);
                अवरोध;
            शेष:
                std::अ_त्रुटि << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::प_पूर्ण;
                GGML_ABORT("fatal error");
            }
        } अन्यथा अगर (tensor->op == GGML_OP_GLU) {
            अगर (src_clone[1] == nullptr) {
                tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } अन्यथा {
                tensor_clone = ggml_glu_split(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
            }
            ggml_set_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
            ggml_set_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
        } अन्यथा अगर (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
            अगर (tensor->src[1] == nullptr) {
                tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } अन्यथा {
                tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } अन्यथा अगर (tensor->op == GGML_OP_CONT) {
            tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } अन्यथा अगर (tensor->op == GGML_OP_RESHAPE) {
            tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } अन्यथा अगर (tensor->op == GGML_OP_VIEW) {
            tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_PERMUTE) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = ggml_permute(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } अन्यथा अगर (tensor->op == GGML_OP_TRANSPOSE) {
            tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_GET_ROWS) {
            tensor_clone = ggml_get_rows(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_ARGSORT) {
            tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_sort_order) *(पूर्णांक *)tensor->op_params);
        } अन्यथा अगर (tensor->op == GGML_OP_शीर्ष_K) {
            tensor_clone = ggml_शीर्ष_k(ggml_ctx, src_clone[0], tensor->ne[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_योग) {
            tensor_clone = ggml_योग(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_योग_ROWS) {
            tensor_clone = ggml_योग_rows(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_CUMSUM) {
            tensor_clone = ggml_cumsum(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_DSV4_HC_COMB) {
            tensor_clone = ggml_dsv4_hc_comb(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                ggml_get_op_params_f32(tensor, 0), ggml_get_op_params_i32(tensor, 1));
        } अन्यथा अगर (tensor->op == GGML_OP_DSV4_HC_PRE) {
            अगर (ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = ggml_dsv4_hc_pre_gated(ggml_ctx, src_clone[0], src_clone[1], ggml_get_op_params_f32(tensor, 0));
            } अन्यथा {
                tensor_clone = ggml_dsv4_hc_pre(ggml_ctx, src_clone[0], src_clone[1]);
            }
        } अन्यथा अगर (tensor->op == GGML_OP_DSV4_HC_POST) {
            tensor_clone = ggml_dsv4_hc_post(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } अन्यथा अगर (tensor->op == GGML_OP_MEAN) {
            tensor_clone = ggml_mean(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_ARGMAX) {
            tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = ggml_cross_entropy_loss(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_CROSS_ENTROPY_LOSS_BACK) {
            tensor_clone = ggml_cross_entropy_loss_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } अन्यथा अगर (tensor->op == GGML_OP_गिनती_EQUAL) {
            tensor_clone = ggml_गिनती_equal(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_SOLVE_TRI) {
            tensor_clone = ggml_solve_tri(ggml_ctx, src_clone[0], src_clone[1], true, true, false);
        } अन्यथा अगर (tensor->op == GGML_OP_IM2COL) {
            स्थिर int32_t s0 = tensor->op_params[0];
            स्थिर int32_t s1 = tensor->op_params[1];
            स्थिर int32_t p0 = tensor->op_params[2];
            स्थिर int32_t p1 = tensor->op_params[3];
            स्थिर int32_t d0 = tensor->op_params[4];
            स्थिर int32_t d1 = tensor->op_params[5];

            स्थिर bool is_2D = tensor->op_params[6] == 1;
            tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } अन्यथा अगर (tensor->op == GGML_OP_IM2COL_3D) {
            स्थिर int32_t s0 = tensor->op_params[0];
            स्थिर int32_t s1 = tensor->op_params[1];
            स्थिर int32_t s2 = tensor->op_params[2];
            स्थिर int32_t p0 = tensor->op_params[3];
            स्थिर int32_t p1 = tensor->op_params[4];
            स्थिर int32_t p2 = tensor->op_params[5];
            स्थिर int32_t d0 = tensor->op_params[6];
            स्थिर int32_t d1 = tensor->op_params[7];
            स्थिर int32_t d2 = tensor->op_params[8];
            स्थिर int32_t IC = tensor->op_params[9];

            tensor_clone = ggml_im2col_3d(ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } अन्यथा अगर (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
            स्थिर int32_t dim = tensor->op_params[0];
            स्थिर int32_t max_period = tensor->op_params[1];
            tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
        } अन्यथा अगर (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
            स्थिर int32_t s0 = tensor->op_params[0];
            स्थिर int32_t p0 = tensor->op_params[1];
            स्थिर int32_t d0 = tensor->op_params[2];
            tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } अन्यथा अगर (tensor->op == GGML_OP_COL2IM_1D) {
            स्थिर int32_t stride = tensor->op_params[0];
            स्थिर int32_t oc     = tensor->op_params[1];
            स्थिर int32_t p0     = tensor->op_params[2];
            tensor_clone = ggml_col2im_1d(ggml_ctx, src_clone[0], stride, oc, p0);
        } अन्यथा अगर (tensor->op == GGML_OP_POOL_1D) {
            क्रमागत ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            स्थिर int32_t k0 = tensor->op_params[1];
            स्थिर int32_t s0 = tensor->op_params[2];
            स्थिर int32_t p0 = tensor->op_params[3];

            tensor_clone = ggml_pool_1d(ggml_ctx, src_clone[0], op, k0, s0, p0);
        } अन्यथा अगर (tensor->op == GGML_OP_POOL_2D) {
            क्रमागत ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
            स्थिर int32_t k0 = tensor->op_params[1];
            स्थिर int32_t k1 = tensor->op_params[2];
            स्थिर int32_t s0 = tensor->op_params[3];
            स्थिर int32_t s1 = tensor->op_params[4];
            स्थिर int32_t p0 = tensor->op_params[5];
            स्थिर int32_t p1 = tensor->op_params[6];

            tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } अन्यथा अगर (tensor->op == GGML_OP_CONV_2D) {
            स्थिर int32_t s0 = tensor->op_params[0];
            स्थिर int32_t s1 = tensor->op_params[1];
            स्थिर int32_t p0 = tensor->op_params[2];
            स्थिर int32_t p1 = tensor->op_params[3];
            स्थिर int32_t d0 = tensor->op_params[4];
            स्थिर int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } अन्यथा अगर (tensor->op == GGML_OP_CONV_3D) {
            स्थिर int32_t s0 = tensor->op_params[0];
            स्थिर int32_t s1 = tensor->op_params[1];
            स्थिर int32_t s2 = tensor->op_params[2];
            स्थिर int32_t p0 = tensor->op_params[3];
            स्थिर int32_t p1 = tensor->op_params[4];
            स्थिर int32_t p2 = tensor->op_params[5];
            स्थिर int32_t d0 = tensor->op_params[6];
            स्थिर int32_t d1 = tensor->op_params[7];
            स्थिर int32_t d2 = tensor->op_params[8];
            स्थिर int32_t IC = tensor->op_params[9];
            स्थिर int32_t N  = tensor->op_params[10];
            स्थिर int32_t OC = tensor->op_params[11];
            tensor_clone = ggml_conv_3d_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } अन्यथा अगर (tensor->op == GGML_OP_CONV_2D_DW) {
            स्थिर int32_t s0 = tensor->op_params[0];
            स्थिर int32_t s1 = tensor->op_params[1];
            स्थिर int32_t p0 = tensor->op_params[2];
            स्थिर int32_t p1 = tensor->op_params[3];
            स्थिर int32_t d0 = tensor->op_params[4];
            स्थिर int32_t d1 = tensor->op_params[5];
            tensor_clone = ggml_conv_2d_dw_direct(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } अन्यथा अगर (tensor->op == GGML_OP_CONV_TRANSPOSE_2D) {
            स्थिर int32_t s = tensor->op_params[0];
            tensor_clone = ggml_conv_transpose_2d_p0(ggml_ctx, src_clone[0], src_clone[1], s);
        } अन्यथा अगर (tensor->op == GGML_OP_LEAKY_RELU) {
            स्थिर भग्न * op_params = (स्थिर भग्न *)tensor->op_params;
            tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], false);
        } अन्यथा अगर (tensor->op == GGML_OP_RWKV_WKV6) {
            tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } अन्यथा अगर (tensor->op == GGML_OP_RWKV_WKV7) {
            tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } अन्यथा अगर (tensor->op == GGML_OP_GATED_LINEAR_ATTN) {
            स्थिर भग्न * op_params = (स्थिर भग्न *)tensor->op_params;
            tensor_clone = ggml_gated_linear_attn(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } अन्यथा अगर (tensor->op == GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = ggml_lightning_indexer(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } अन्यथा अगर (tensor->op == GGML_OP_GATED_DELTA_NET) {
            tensor_clone = ggml_gated_delta_net(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            ggml_get_op_params_i32(tensor, 0));
        } अन्यथा अगर (tensor->op == GGML_OP_OPT_STEP_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_adamw(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } अन्यथा अगर (tensor->op == GGML_OP_OPT_STEP_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = ggml_opt_step_sgd(ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } अन्यथा अगर (tensor->op == GGML_OP_ADD_ID) {
            tensor_clone = ggml_add_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } अन्यथा अगर (tensor->op == GGML_OP_SSM_SCAN) {
            स्थिर int32_t K = ggml_get_op_params_i32(tensor, 0);
            tensor_clone = ggml_ssm_scan(ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } अन्यथा अगर (tensor->op == GGML_OP_SSM_CONV) {
            tensor_clone = ggml_ssm_conv(ggml_ctx, src_clone[0], src_clone[1]);
        } अन्यथा अगर (tensor->op == GGML_OP_ROLL) {
            स्थिर int32_t s0 = tensor->op_params[0];
            स्थिर int32_t s1 = tensor->op_params[1];
            स्थिर int32_t s2 = tensor->op_params[2];
            स्थिर int32_t s3 = tensor->op_params[3];
            tensor_clone = ggml_roll(ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        अन्यथा {
            std::अ_त्रुटि << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::प_पूर्ण;
            GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    ggml_cgraph * cgraph_cpu = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    अगर (vk_output_tensor > 0 && vk_output_tensor == check_गणक) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_आकार = ggml_nbytes(tensor_clone);

    comp_परिणाम = दो_स्मृति(comp_आकार);
    स_नकल(comp_परिणाम, tensor_clone->तथ्य, comp_आकार);
    स_नकल(comp_nb, tensor_clone->nb, माप(माप_प्रकार) * GGML_MAX_DIMS);

    क्रम (स्वतः m : cloned_mallocs) {
        मुक्त(m);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

व्योम ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, पूर्णांक tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->अंक_additional_fused_ops];
    अगर (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        वापस;
    }

    अगर (!(vk_output_tensor > 0 && vk_output_tensor == check_गणक) && check_गणक <= vk_skip_checks) {
        वापस;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    व्योम * tensor_तथ्य = tensor->तथ्य;

    अगर (ggml_backend_बफर_is_vk(tensor->बफर)) {
        माप_प्रकार tensor_आकार = ggml_nbytes(tensor);
        tensor_तथ्य = दो_स्मृति(tensor_आकार);

        ggml_backend_vk_बफर_context * buf_ctx = (ggml_backend_vk_बफर_context *)tensor->बफर->context;

        vk_बफर& बफर_gpu = buf_ctx->dev_बफर;
        uint64_t offset = vk_tensor_offset(tensor) + tensor->view_offs;
        अगर (offset + tensor_आकार >= बफर_gpu->size) {
            tensor_आकार = बफर_gpu->size - offset;
        }

        ggml_vk_बफर_read(बफर_gpu, offset, tensor_तथ्य, tensor_आकार);
    }

    भग्न first_error_परिणाम = -1.0f;
    भग्न first_error_correct = -1.0f;
    std::array<पूर्णांक, 4> first_error = { -1, -1, -1, -1 };
    द्विगुन avg_err = 0.0;
    माप_प्रकार गणक = 0;

    क्रम (पूर्णांक i3 = 0; i3 < tensor->ne[3]; i3++) {
        क्रम (पूर्णांक i2 = 0; i2 < tensor->ne[2]; i2++) {
            क्रम (पूर्णांक i1 = 0; i1 < tensor->ne[1]; i1++) {
                क्रम (पूर्णांक i0 = 0; i0 < tensor->ne[0]; i0++) {
                    स्थिर bool बफर_आकार_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_आकार;
                    भग्न correct = 0.0f;
                    भग्न परिणाम = 0.0f;

                    अगर (बफर_आकार_fit) {
                        अगर (tensor->type == GGML_TYPE_F32) {
                            correct = *(भग्न *) ((अक्षर *) comp_परिणाम + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            परिणाम  = *(भग्न *) ((अक्षर *) tensor_तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } अन्यथा अगर (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((अक्षर *) comp_परिणाम + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            परिणाम  = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((अक्षर *) tensor_तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } अन्यथा अगर (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((अक्षर *) comp_परिणाम + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            परिणाम  = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((अक्षर *) tensor_तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } अन्यथा अगर (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((अक्षर *) comp_परिणाम + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            परिणाम  = *(int32_t *) ((अक्षर *) tensor_तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } अन्यथा अगर (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((अक्षर *) comp_परिणाम + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            परिणाम  = *(int64_t *) ((अक्षर *) tensor_तथ्य + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } अन्यथा {
                            std::अ_त्रुटि << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::प_पूर्ण;
                        }
                    } अन्यथा {
                        std::अ_त्रुटि << "Missing debug code for type " << ggml_type_name(tensor->type) << std::प_पूर्ण;
                        GGML_ABORT("fatal error");
                    }

                    अगर ((std::isnan(correct) != std::isnan(परिणाम)) || (std::isinf(correct) != std::isinf(परिणाम)) || !बफर_आकार_fit) {
                        std::अ_त्रुटि << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << परिणाम << " correct=" << correct << " avg_err=" << (avg_err / गणक) << std::प_पूर्ण;
                        std::अ_त्रुटि << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::प_पूर्ण;
                        अगर (src0 != nullptr) {
                            std::अ_त्रुटि << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::प_पूर्ण;
                        }
                        अगर (src1 != nullptr) {
                            std::अ_त्रुटि << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::प_पूर्ण;
                        }
                        अगर (src2 != nullptr) {
                            std::अ_त्रुटि << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::प_पूर्ण;
                        }
                        अगर (src3 != nullptr) {
                            std::अ_त्रुटि << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::प_पूर्ण;
                        }
                        std::अ_त्रुटि << "First error: result=" << first_error_परिणाम << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::प_पूर्ण;
                        std::अ_त्रुटि << std::प_पूर्ण << "Result:" << std::प_पूर्ण;
                        ggml_vk_print_tensor_area(tensor, tensor_तथ्य, i0, i1, i2, i3);
                        std::अ_त्रुटि << std::प_पूर्ण << "Correct:" << std::प_पूर्ण;
                        ggml_vk_print_tensor_area(tensor, comp_परिणाम, i0, i1, i2, i3);
                        std::अ_त्रुटि << std::प_पूर्ण;
                        std::vector<स्थिर ggml_tensor *> done;
                        ggml_vk_print_graph_origin(tensor, done);
                        GGML_ABORT("fatal error");
                    }
                    स्थिर द्विगुन denom = std::भ_असल(correct) > 1.0f ? (std::भ_असल(correct) > 1e-8 ? std::भ_असल(correct) : 1e-8) : 1.0f;
                    अगर (first_error[0] == -1 && std::भ_असल(correct - परिणाम) / denom > 0.5) {
                        first_error[0] = i0;
                        first_error[1] = i1;
                        first_error[2] = i2;
                        first_error[3] = i3;
                        first_error_परिणाम = परिणाम;
                        first_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    अगर (!std::isinf(correct) && !std::isinf(परिणाम) && !std::isnan(correct) && !std::isnan(परिणाम)) {
                        avg_err += std::भ_असल(correct - परिणाम) / denom;
                    }
                    गणक++;
                }
            }
        }
    }

    avg_err /= गणक;

    अगर (vk_output_tensor > 0 && vk_output_tensor == check_गणक) {
        std::अ_त्रुटि << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_गणक << ")" << std::प_पूर्ण;
        std::अ_त्रुटि << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::प_पूर्ण;
        अगर (src0 != nullptr) {
            std::अ_त्रुटि << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::प_पूर्ण;
        }
        अगर (src1 != nullptr) {
            std::अ_त्रुटि << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::प_पूर्ण;
        }
        अगर (src2 != nullptr) {
            std::अ_त्रुटि << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::प_पूर्ण;
        }
        अगर (src3 != nullptr) {
            std::अ_त्रुटि << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::प_पूर्ण;
        }
        std::अ_त्रुटि << "First error: result=" << first_error_परिणाम << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::प_पूर्ण;
        std::अ_त्रुटि << std::प_पूर्ण << "Result:" << std::प_पूर्ण;
        ggml_vk_print_tensor_area(tensor, tensor_तथ्य, 5, 5, 0, 0);
        std::अ_त्रुटि << std::प_पूर्ण << "Correct:" << std::प_पूर्ण;
        ggml_vk_print_tensor_area(tensor, comp_परिणाम, 5, 5, 0, 0);
        std::अ_त्रुटि << std::प_पूर्ण;
        std::vector<स्थिर ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
    }

    अगर (avg_err > 0.01 || std::isnan(avg_err)) {
        std::अ_त्रुटि << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_गणक << ")" << std::प_पूर्ण;
        std::अ_त्रुटि << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::प_पूर्ण;
        अगर (src0 != nullptr) {
            std::अ_त्रुटि << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::प_पूर्ण;
        }
        अगर (src1 != nullptr) {
            std::अ_त्रुटि << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::प_पूर्ण;
        }
        अगर (src2 != nullptr) {
            std::अ_त्रुटि << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::प_पूर्ण;
        }
        अगर (src3 != nullptr) {
            std::अ_त्रुटि << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::प_पूर्ण;
        }
        std::अ_त्रुटि << "First error: result=" << first_error_परिणाम << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::प_पूर्ण;
        std::अ_त्रुटि << std::प_पूर्ण << "Result:" << std::प_पूर्ण;
        ggml_vk_print_tensor_area(tensor, tensor_तथ्य, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::अ_त्रुटि << std::प_पूर्ण << "Correct:" << std::प_पूर्ण;
        ggml_vk_print_tensor_area(tensor, comp_परिणाम, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::अ_त्रुटि << std::प_पूर्ण;
        std::vector<स्थिर ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
        GGML_ABORT("fatal error");
    } अन्यथा {
        std::अ_त्रुटि << check_गणक << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::प_पूर्ण;
    }

    मुक्त(comp_परिणाम);
    comp_परिणाम = nullptr;
    comp_आकार = 0;

    अगर (ggml_backend_बफर_is_vk(tensor->बफर)) {
        मुक्त(tensor_तथ्य);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#पूर्ण_अगर

