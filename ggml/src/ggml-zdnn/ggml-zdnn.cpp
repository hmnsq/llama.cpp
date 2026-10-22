#अन्तर्भाव "ggml-zdnn.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव "ggml-zdnn/common.hpp"
#अन्तर्भाव "ggml-zdnn/fattn.hpp"
#अन्तर्भाव "ggml-zdnn/mmf.hpp"
#अन्तर्भाव "ggml-zdnn/mmq.hpp"
#अन्तर्भाव "ggml-zdnn/utils.hpp"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <vector>
#अन्तर्भाव <memory>
#अन्तर्भाव <csignal>  // raise(SIGTRAP)
#अन्तर्भाव <unistd.h>

स्थिर शून्य ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];  // weights
    ध्रुव ggml_tensor * src1 = dst->src[1];  // inputs

    यदि (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } अन्यथा {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

स्थिर तार्किक ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    विकल्प (dst->op) {
        पक्ष GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } भङ्ग;

        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } भङ्ग;

        स्वभाव:
            प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

स्थिर गणना ggml_स्थितिः ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    प्रति (पूर्णाङ्क i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * ग्रन्थिः = gf->nodes[i];

        यदि (ggml_is_रिक्तानि(ग्रन्थिः)
            || ग्रन्थिः->op == GGML_OP_NONE
            || ग्रन्थिः->op == GGML_OP_RESHAPE
            || ग्रन्थिः->op == GGML_OP_VIEW
            || ग्रन्थिः->op == GGML_OP_PERMUTE
            || ग्रन्थिः->op == GGML_OP_TRANSPOSE) {
            अनुवर्त;
        }

        यदि ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            अनुवर्त;
        }

        तार्किक ok = ggml_zdnn_compute_forward(ctx, ग्रन्थिः);
        यदि (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, ग्रन्थिः->name, ggml_op_name(ग्रन्थिः->op));
        }

        GGML_ASSERT(ok);
    }

    प्रत्यागम GGML_स्थितिः_SUCCESS;

    GGML_अप्रयुक्तम्(ctx_dev);
}

स्थिर तार्किक ggml_zdnn_supports_op(ध्रुव ggml_backend_zdnn_device_context * ctx_dev, ध्रुव ggml_tensor * op) {
    विकल्प (op->op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_PERMUTE:
            प्रत्यागम सत्य;

        पक्ष GGML_OP_MUL_MAT:
            {
                ध्रुव ggml_tensor * weights = op->src[0];
                ध्रुव ggml_tensor * inputs  = op->src[1];

                ध्रुव int64_t ne10 = inputs->ne[0];
                ध्रुव int64_t ne0  = op->ne[0];
                ध्रुव int64_t ne1  = op->ne[1];

                ध्रुव int64_t max_batch = ctx_dev->max_size;

                यदि (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != शून्यसूचकः ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        प्रत्यागम असत्य;
                }

                विकल्प (weights->type) {
                    पक्ष GGML_TYPE_F32:
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        प्रत्यागम inputs->view_src == शून्यसूचकः;
                    पक्ष GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        प्रत्यागम ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    स्वभाव:
                        प्रत्यागम असत्य;
                }
            } भङ्ग;

        पक्ष GGML_OP_FLASH_ATTN_EXT:
            प्रत्यागम ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        स्वभाव:
            प्रत्यागम असत्य;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
स्थिर ggml_backend_reg    g_ggml_backend_zdnn_reg;
स्थिर ggml_backend_device g_ggml_backend_zdnn_device;

स्थिर ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ असत्य,
    /* .has_parmblkformat_1   = */ असत्य,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

स्थिर पूर्णाङ्क ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    प्रतिज्ञा(ctx != रिक्त);

    यदि (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    यदि (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_आकार();
        strncpy(ctx->name, GGML_ZDNN_NAME, परिमाण(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_गणनम्++;
    प्रत्यागम ctx->zdnn_device;
}

स्थिर शून्य ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    प्रतिज्ञा(ctx != रिक्त);
    प्रतिज्ञा(ctx->zdnn_device_ref_गणनम् > 0);

    ctx->zdnn_device_ref_गणनम्--;
    यदि (ctx->zdnn_device_ref_गणनम् == 0) {
        यदि (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

स्थिर ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #यदि_परिभाषितम् STATIC_LIB
    zdnn_init();
    #यदि_अन्तः

    ggml_backend_zdnn_context * ctx = नूतन ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    पूर्णाङ्क device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_तन्तवः = GGML_DEFAULT_N_तन्तवः;
    ctx->gf = शून्यसूचकः;

    प्रत्यागम ctx;
}

स्थिर शून्य ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    प्रति (ध्रुव स्वचालित & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_बफर(&it.second->ztensor));
    }

    विलोपय ctx;
}

//
// backend interface
//

स्थिर शून्य ggml_backend_zdnn_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    प्रति (ध्रुव स्वचालित & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_बफर * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        यदि (buf->extra != शून्यसूचकः) मुक्त(buf->extra->दत्त);
        यदि (buf->ztensor.बफर_आकार > 0) ZDNN_CHECK(zdnn_free_ztensor_बफर(&buf->ztensor));
    }

    यदि (ctx->owned) {
        ggml_aligned_free(ctx->all_दत्त, ctx->all_आकार);
    }

    विलोपय ctx;
}

स्थिर शून्य * ggml_backend_zdnn_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;
    प्रत्यागम ctx->all_दत्त;
}

स्थिर गणना ggml_स्थितिः ggml_backend_zdnn_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    यदि (tensor->view_src != रिक्त) {
        प्रतिज्ञा(tensor->view_src->बफर->buft == बफर->buft);
        प्रत्यागम GGML_स्थितिः_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    यदि (ggml_is_रिक्तानि(tensor)) {
        प्रत्यागम GGML_स्थितिः_SUCCESS;
    }

    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    ध्रुव int64_t tsize = ggml_nbytes(tensor);
    पूर्णाङ्क बफर_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_बफर> zdnn_बफर = std::make_unique<ggml_backend_zdnn_बफर>();
    zdnn_बफर->दत्त = tensor->दत्त;
    zdnn_बफर->size = tsize;
    zdnn_बफर->extra = शून्यसूचकः;
    snprintf(zdnn_बफर->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_बफर.get(), tensor);
    tensor->extra = zdnn_बफर.get();

    विकल्प (tensor->op) {
        पक्ष GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_बफर> zdnn_bias_बफर = std::make_unique<ggml_backend_zdnn_बफर>();
                zdnn_bias_बफर->दत्त = (शून्य *)शून्यस्मृतिं_देहि(tensor->ne[0], ggml_element_आकार(tensor));
                zdnn_bias_बफर->size = ggml_element_आकार(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_बफर->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (पूर्णाङ्क)परिमाण(" (bias)"), tensor->name);

                ध्रुव int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_बफर->pre_tfm_desc,
                                        zdnn_bias_बफर->tfm_desc,
                                        zdnn_bias_बफर->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_आरोपय_tensor(zdnn_bias_बफर->ztensor, zdnn_bias_बफर->दत्त);
                zdnn_बफर->extra = zdnn_bias_बफर.get();

                ctx->buffers.push_back(std::move(zdnn_bias_बफर));
                ctx->n_buffers++;
            } भङ्ग;
        स्वभाव:
            भङ्ग;
    }

    ctx->buffers.push_back(std::move(zdnn_बफर));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    प्रत्यागम GGML_स्थितिः_SUCCESS;

    GGML_अप्रयुक्तम्(बफर_idx);
}

स्थिर शून्य ggml_backend_zdnn_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, size_t विस्थापनम्, size_t size) {
    स्मृतिं_पूरय((वर्ण *)tensor->दत्त + विस्थापनम्, value, size);

    GGML_अप्रयुक्तम्(बफर);
}

स्थिर शून्य ggml_backend_zdnn_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    स्मृतिं_प्रतिलिख((वर्ण *)tensor->दत्त + विस्थापनम्, दत्त, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    यदि (tensor->view_src != शून्यसूचकः) {
        ggml_backend_zdnn_बफर * src_extra = (ggml_backend_zdnn_बफर *)tensor->view_src->extra;
        यदि (src_extra != शून्यसूचकः && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        प्रत्यागम;
    }

    ggml_backend_zdnn_बफर * extra = (ggml_backend_zdnn_बफर *)tensor->extra;
    यदि (extra == शून्यसूचकः) {
        प्रत्यागम;
    }

    यदि (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(विस्थापनम् == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_आरोपय_tensor_q(extra, tensor);
        प्रत्यागम;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    यदि (tensor->बफर->usage != GGML_BACKEND_बफर_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        प्रत्यागम;
    }
    यदि (extra->ztensor.is_transformed == असत्य) ggml_zdnn_आरोपय_tensor(extra->ztensor, tensor->दत्त);

    GGML_अप्रयुक्तम्(बफर);
}

स्थिर शून्य ggml_backend_zdnn_बफर_get_tensor(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    स्मृतिं_प्रतिलिख(दत्त, (ध्रुव वर्ण *)tensor->दत्त + विस्थापनम्, size);

    GGML_अप्रयुक्तम्(बफर);
}

स्थिर शून्य ggml_backend_zdnn_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    स्मृतिं_पूरय(ctx->all_दत्त, value, ctx->all_आकार);
}

स्थिर शून्य ggml_backend_zdnn_बफर_reset(ggml_backend_बफर_t बफर) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    प्रति (ध्रुव स्वचालित & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_बफर * buf = buf_ptr.get();

        यदि (buf->extra != शून्यसूचकः) {
            मुक्त(buf->extra->दत्त);
        }

        यदि (buf->ztensor.बफर_आकार > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_बफर(&buf->ztensor));
        }
    }

    यदि (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

स्थिर ggml_backend_बफर_i ggml_backend_zdnn_बफर_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_बफर_free_बफर,
    /* .get_base      = */ ggml_backend_zdnn_बफर_get_आधारः,
    /* .init_tensor   = */ ggml_backend_zdnn_बफर_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_बफर_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_बफर_समुच्चयः_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_बफर_get_tensor,
    /* .set_tensor_2d = */ रिक्त,
    /* .get_tensor_2d = */ रिक्त,
    /* .cpy_tensor    = */ रिक्त,
    /* .clear         = */ ggml_backend_zdnn_बफर_clear,
    /* .reset         = */ ggml_backend_zdnn_बफर_reset,
};

//
// default buffer type
//

स्थिर ध्रुव वर्ण * ggml_backend_zdnn_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    प्रत्यागम GGML_ZDNN_NAME;

    GGML_अप्रयुक्तम्(buft);
}

स्थिर ggml_backend_बफर_t ggml_backend_zdnn_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    ggml_backend_zdnn_बफर_context * ctx = नूतन ggml_backend_zdnn_बफर_context();

    ध्रुव size_t आकार_page = sysconf(_SC_PAGESIZE);

    size_t आकार_aligned = size;
    यदि ((आकार_aligned % आकार_page) != 0) {
        आकार_aligned += आकार_page - (आकार_aligned % आकार_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    पूर्णाङ्क device = ctx_dev->zdnn_device; GGML_अप्रयुक्तम्(device);

    ctx->all_दत्त  = ggml_aligned_malloc(आकार_aligned);
    ctx->all_आकार  = आकार_aligned;
    ctx->owned     = सत्य;
    ctx->n_buffers = 1;

    यदि (ctx->all_दत्त != रिक्त) {
        std::unique_ptr<ggml_backend_zdnn_बफर> zdnn_बफर = std::make_unique<ggml_backend_zdnn_बफर>();
        zdnn_बफर->दत्त = ctx->all_दत्त;
        zdnn_बफर->size = आकार_aligned;
        ctx->buffers.push_back(std::move(zdnn_बफर));
    }

    यदि (आकार_aligned > 0 && (ctx->all_दत्त == रिक्त)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, आकार_aligned / 1024.0 / 1024.0);
        विलोपय ctx;
        प्रत्यागम रिक्त;
    }

    प्रत्यागम ggml_backend_बफर_init(buft, ggml_backend_zdnn_बफर_i, ctx, size);
}

स्थिर size_t ggml_backend_zdnn_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    प्रत्यागम 256;

    GGML_अप्रयुक्तम्(buft);
}

स्थिर तार्किक ggml_backend_zdnn_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    प्रत्यागम असत्य;

    GGML_अप्रयुक्तम्(buft);
}

ggml_backend_बफर_type_t ggml_backend_zdnn_बफर_type(शून्य) {
    स्थिर ggml_backend_बफर_type ggml_backend_बफर_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_बफर_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_बफर_type_alloc_बफर,
            /* .alloc_buffer_n      = */ रिक्त,
            /* .get_alignment       = */ ggml_backend_zdnn_बफर_type_get_alignment,
            /* .get_max_size        = */ रिक्त,
            /* .get_alloc_size      = */ रिक्त,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ रिक्त,
            /* .is_host             = */ ggml_backend_zdnn_बफर_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ रिक्त,
    };

    प्रत्यागम &ggml_backend_बफर_type_zdnn;
}

//
// backend
//

स्थिर ध्रुव वर्ण * ggml_backend_zdnn_name(ggml_backend_t backend) {
    प्रत्यागम GGML_ZDNN_NAME;

    GGML_अप्रयुक्तम्(backend);
}

स्थिर शून्य ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    मुक्त(backend);
}

स्थिर गणना ggml_स्थितिः ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    प्रत्यागम ggml_zdnn_graph_compute(backend, cgraph);
}

स्थिर ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ रिक्त,
    /* .get_tensor_async       = */ रिक्त,
    /* .set_tensor_2d_async    = */ रिक्त,
    /* .get_tensor_2d_async    = */ रिक्त,
    /* .cpy_tensor_async       = */ रिक्त,
    /* .synchronize            = */ रिक्त,
    /* .graph_plan_create      = */ रिक्त,
    /* .graph_plan_free        = */ रिक्त,
    /* .graph_plan_update      = */ रिक्त,
    /* .graph_plan_compute     = */ रिक्त,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ रिक्त,
    /* .event_wait             = */ रिक्त,
    /* .graph_optimize         = */ रिक्त,
};

स्थिर ggml_guid_t ggml_backend_zdnn_guid(शून्य) {
    स्थिर वर्ण guid_str[] = "IBM-ZDNN-ACCELER";
    प्रत्यागम reinterpret_cast<ggml_guid_t>(guid_str);
}

तार्किक ggml_backend_is_zdnn(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

स्थिर ध्रुव वर्ण * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    प्रत्यागम GGML_ZDNN_NAME;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ध्रुव वर्ण * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    प्रत्यागम "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_अप्रयुक्तम्(dev);
}

स्थिर शून्य ggml_backend_zdnn_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
    *मुक्त  = 0;
    *योग = 0;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर शून्य ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);
    props->caps = {
        /* .async                = */ असत्य,
        /* .host_buffer          = */ असत्य,
        /* .buffer_from_host_ptr = */ असत्य,
        /* .events               = */ असत्य,
        /* .mmap_support         = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    यदि (ctx == रिक्त) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        प्रत्यागम रिक्त;
    }

    ggml_backend_t backend = (ggml_backend *)स्मृतिं_देहि(परिमाण(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    प्रत्यागम backend;

    GGML_अप्रयुक्तम्(params);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_zdnn_device_get_बफर_type(ggml_backend_dev_t dev) {
    प्रत्यागम ggml_backend_zdnn_बफर_type();

    GGML_अप्रयुक्तम्(dev);
}

स्थिर तार्किक ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    प्रत्यागम ggml_zdnn_supports_op(ctx_dev, op);
}

स्थिर तार्किक ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    प्रत्यागम
        buft->iface.get_name == ggml_backend_zdnn_बफर_type_get_name;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_स्मृतिः,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_बफर_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ रिक्त,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ रिक्त,
    /* .event_new            = */ रिक्त,
    /* .event_free           = */ रिक्त,
    /* .event_synchronize    = */ रिक्त,
};

//
// backend registry
//

स्थिर ध्रुव वर्ण * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    प्रत्यागम GGML_ZDNN_NAME;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर size_t ggml_backend_zdnn_reg_device_गणनम्(ggml_backend_reg_t reg) {
    यदि (!zdnn_is_nnpa_installed()) {
        प्रत्यागम 0;
    }
    प्रत्यागम 1;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);

    प्रत्यागम &g_ggml_backend_zdnn_device;

    GGML_अप्रयुक्तम्(reg);
    GGML_अप्रयुक्तम्(सूचकांक);
}

स्थिर ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { रिक्त, रिक्त },
};

स्थिर ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    प्रत्यागम g_ggml_backend_zdnn_features;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर शून्य ggml_backend_zdnn_समुच्चयः_n_तन्तवः(ggml_backend_t backend, पूर्णाङ्क n_तन्तवः) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_तन्तवः = n_तन्तवः;
}

स्थिर शून्य * ggml_backend_zdnn_get_proc_स्थानाङ्कः(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    यदि (वाक्यं_तुलय(name, "ggml_backend_get_features") == 0) {
        प्रत्यागम (शून्य *) ggml_backend_zdnn_get_features;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_set_n_threads") == 0) {
        प्रत्यागम (शून्य *) ggml_backend_zdnn_समुच्चयः_n_तन्तवः;
    }

    प्रत्यागम रिक्त;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_गणनम्,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_स्थानाङ्कः
};

स्थिर शून्य ggml_zdnn_cleanup(शून्य) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(शून्य) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_संस्करणम्,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ रिक्त
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        प्रत्यागम &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
