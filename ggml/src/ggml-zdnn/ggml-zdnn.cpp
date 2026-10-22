#అన్తర్భావ "ggml-zdnn.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ "ggml-zdnn/common.hpp"
#అన్తర్భావ "ggml-zdnn/fattn.hpp"
#అన్తర్భావ "ggml-zdnn/mmf.hpp"
#అన్తర్భావ "ggml-zdnn/mmq.hpp"
#అన్తర్భావ "ggml-zdnn/utils.hpp"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <vector>
#అన్తర్భావ <memory>
#అన్తర్భావ <csignal>  // raise(SIGTRAP)
#అన్తర్భావ <unistd.h>

స్థిర శూన్య ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];  // weights
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];  // inputs

    ఒకవేళ (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } లేకపోతే {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

స్థిర తార్కిక ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    ఎంపిక (dst->op) {
        స్థితి GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } ఆపు;

        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } ఆపు;

        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర గణన ggml_స్థితి ggml_zdnn_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    కొరకు (పూర్ణాంకం i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * కణుపు = gf->nodes[i];

        ఒకవేళ (ggml_is_ఖాళీలు(కణుపు)
            || కణుపు->op == GGML_OP_NONE
            || కణుపు->op == GGML_OP_RESHAPE
            || కణుపు->op == GGML_OP_VIEW
            || కణుపు->op == GGML_OP_క్రమమార్పు
            || కణుపు->op == GGML_OP_TRANSPOSE) {
            కొనసాగు;
        }

        ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            కొనసాగు;
        }

        తార్కిక ok = ggml_zdnn_compute_forward(ctx, కణుపు);
        ఒకవేళ (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, కణుపు->name, ggml_op_name(కణుపు->op));
        }

        GGML_ASSERT(ok);
    }

    తిరిగి GGML_స్థితి_SUCCESS;

    GGML_వాడని(ctx_dev);
}

స్థిర తార్కిక ggml_zdnn_supports_op(స్థిరరాశి ggml_backend_zdnn_device_context * ctx_dev, స్థిరరాశి ggml_tensor * op) {
    ఎంపిక (op->op) {
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_క్రమమార్పు:
            తిరిగి నిజం;

        స్థితి GGML_OP_MUL_MAT:
            {
                స్థిరరాశి ggml_tensor * weights = op->src[0];
                స్థిరరాశి ggml_tensor * inputs  = op->src[1];

                స్థిరరాశి int64_t ne10 = inputs->ne[0];
                స్థిరరాశి int64_t ne0  = op->ne[0];
                స్థిరరాశి int64_t ne1  = op->ne[1];

                స్థిరరాశి int64_t max_batch = ctx_dev->max_size;

                ఒకవేళ (!ggml_is_మాతృక(weights) || !ggml_is_మాతృక(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != శూన్యసూచిక ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        తిరిగి అబద్ధం;
                }

                ఎంపిక (weights->type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        తిరిగి inputs->view_src == శూన్యసూచిక;
                    స్థితి GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        తిరిగి ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            } ఆపు;

        స్థితి GGML_OP_FLASH_ATTN_EXT:
            తిరిగి ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
స్థిర ggml_backend_reg    g_ggml_backend_zdnn_reg;
స్థిర ggml_backend_device g_ggml_backend_zdnn_device;

స్థిర ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ అబద్ధం,
    /* .has_parmblkformat_1   = */ అబద్ధం,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

స్థిర పూర్ణాంకం ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    ప్రతిజ్ఞా(ctx != ఖాళీ);

    ఒకవేళ (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    ఒకవేళ (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_కొలత();
        strncpy(ctx->name, GGML_ZDNN_NAME, పరిమాణం(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_లెక్క++;
    తిరిగి ctx->zdnn_device;
}

స్థిర శూన్య ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    ప్రతిజ్ఞా(ctx != ఖాళీ);
    ప్రతిజ్ఞా(ctx->zdnn_device_ref_లెక్క > 0);

    ctx->zdnn_device_ref_లెక్క--;
    ఒకవేళ (ctx->zdnn_device_ref_లెక్క == 0) {
        ఒకవేళ (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

స్థిర ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #నిర్వచితమైతే STATIC_LIB
    zdnn_init();
    #అయితే_ముగింపు

    ggml_backend_zdnn_context * ctx = కొత్త ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    పూర్ణాంకం device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_దారాలు = GGML_DEFAULT_N_దారాలు;
    ctx->gf = శూన్యసూచిక;

    తిరిగి ctx;
}

స్థిర శూన్య ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    కొరకు (స్థిరరాశి స్వయంచాలక & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_బఫరు(&it.second->ztensor));
    }

    తొలగించు ctx;
}

//
// backend interface
//

స్థిర శూన్య ggml_backend_zdnn_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_zdnn_బఫరు_context * ctx = (ggml_backend_zdnn_బఫరు_context *)బఫరు->context;

    కొరకు (స్థిరరాశి స్వయంచాలక & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_బఫరు * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        ఒకవేళ (buf->extra != శూన్యసూచిక) విడుదల(buf->extra->దత్తాంశం);
        ఒకవేళ (buf->ztensor.బఫరు_కొలత > 0) ZDNN_CHECK(zdnn_free_ztensor_బఫరు(&buf->ztensor));
    }

    ఒకవేళ (ctx->owned) {
        ggml_aligned_free(ctx->all_దత్తాంశం, ctx->all_కొలత);
    }

    తొలగించు ctx;
}

స్థిర శూన్య * ggml_backend_zdnn_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_zdnn_బఫరు_context * ctx = (ggml_backend_zdnn_బఫరు_context *)బఫరు->context;
    తిరిగి ctx->all_దత్తాంశం;
}

స్థిర గణన ggml_స్థితి ggml_backend_zdnn_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor) {
    ఒకవేళ (tensor->view_src != ఖాళీ) {
        ప్రతిజ్ఞా(tensor->view_src->బఫరు->buft == బఫరు->buft);
        తిరిగి GGML_స్థితి_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    ఒకవేళ (ggml_is_ఖాళీలు(tensor)) {
        తిరిగి GGML_స్థితి_SUCCESS;
    }

    ggml_backend_zdnn_బఫరు_context * ctx = (ggml_backend_zdnn_బఫరు_context *)బఫరు->context;

    స్థిరరాశి int64_t tsize = ggml_nbytes(tensor);
    పూర్ణాంకం బఫరు_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_బఫరు> zdnn_బఫరు = std::make_unique<ggml_backend_zdnn_బఫరు>();
    zdnn_బఫరు->దత్తాంశం = tensor->దత్తాంశం;
    zdnn_బఫరు->size = tsize;
    zdnn_బఫరు->extra = శూన్యసూచిక;
    snprintf(zdnn_బఫరు->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_బఫరు.get(), tensor);
    tensor->extra = zdnn_బఫరు.get();

    ఎంపిక (tensor->op) {
        స్థితి GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_బఫరు> zdnn_bias_బఫరు = std::make_unique<ggml_backend_zdnn_బఫరు>();
                zdnn_bias_బఫరు->దత్తాంశం = (శూన్య *)సున్నా_జ్ఞాపకం_ఇవ్వు(tensor->ne[0], ggml_element_కొలత(tensor));
                zdnn_bias_బఫరు->size = ggml_element_కొలత(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_బఫరు->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (పూర్ణాంకం)పరిమాణం(" (bias)"), tensor->name);

                స్థిరరాశి int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_బఫరు->pre_tfm_desc,
                                        zdnn_bias_బఫరు->tfm_desc,
                                        zdnn_bias_బఫరు->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_ఎక్కించు_tensor(zdnn_bias_బఫరు->ztensor, zdnn_bias_బఫరు->దత్తాంశం);
                zdnn_బఫరు->extra = zdnn_bias_బఫరు.get();

                ctx->buffers.push_back(std::move(zdnn_bias_బఫరు));
                ctx->n_buffers++;
            } ఆపు;
        అప్రమేయ:
            ఆపు;
    }

    ctx->buffers.push_back(std::move(zdnn_బఫరు));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    తిరిగి GGML_స్థితి_SUCCESS;

    GGML_వాడని(బఫరు_idx);
}

స్థిర శూన్య ggml_backend_zdnn_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    memset((అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, value, size);

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_zdnn_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    జ్ఞాపకం_నకలు((అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    ఒకవేళ (tensor->view_src != శూన్యసూచిక) {
        ggml_backend_zdnn_బఫరు * src_extra = (ggml_backend_zdnn_బఫరు *)tensor->view_src->extra;
        ఒకవేళ (src_extra != శూన్యసూచిక && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        తిరిగి;
    }

    ggml_backend_zdnn_బఫరు * extra = (ggml_backend_zdnn_బఫరు *)tensor->extra;
    ఒకవేళ (extra == శూన్యసూచిక) {
        తిరిగి;
    }

    ఒకవేళ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(స్థానభ్రంశం == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_ఎక్కించు_tensor_q(extra, tensor);
        తిరిగి;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    ఒకవేళ (tensor->బఫరు->usage != GGML_BACKEND_బఫరు_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        తిరిగి;
    }
    ఒకవేళ (extra->ztensor.is_transformed == అబద్ధం) ggml_zdnn_ఎక్కించు_tensor(extra->ztensor, tensor->దత్తాంశం);

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_zdnn_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, size);

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_zdnn_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    ggml_backend_zdnn_బఫరు_context * ctx = (ggml_backend_zdnn_బఫరు_context *)బఫరు->context;

    memset(ctx->all_దత్తాంశం, value, ctx->all_కొలత);
}

స్థిర శూన్య ggml_backend_zdnn_బఫరు_reset(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_zdnn_బఫరు_context * ctx = (ggml_backend_zdnn_బఫరు_context *)బఫరు->context;

    కొరకు (స్థిరరాశి స్వయంచాలక & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_బఫరు * buf = buf_ptr.get();

        ఒకవేళ (buf->extra != శూన్యసూచిక) {
            విడుదల(buf->extra->దత్తాంశం);
        }

        ఒకవేళ (buf->ztensor.బఫరు_కొలత > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_బఫరు(&buf->ztensor));
        }
    }

    ఒకవేళ (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

స్థిర ggml_backend_బఫరు_i ggml_backend_zdnn_బఫరు_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_బఫరు_free_బఫరు,
    /* .get_base      = */ ggml_backend_zdnn_బఫరు_get_పునాది,
    /* .init_tensor   = */ ggml_backend_zdnn_బఫరు_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_బఫరు_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_బఫరు_సమితి_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_బఫరు_get_tensor,
    /* .set_tensor_2d = */ ఖాళీ,
    /* .get_tensor_2d = */ ఖాళీ,
    /* .cpy_tensor    = */ ఖాళీ,
    /* .clear         = */ ggml_backend_zdnn_బఫరు_clear,
    /* .reset         = */ ggml_backend_zdnn_బఫరు_reset,
};

//
// default buffer type
//

స్థిర స్థిరరాశి అక్షరం * ggml_backend_zdnn_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి GGML_ZDNN_NAME;

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_zdnn_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    ggml_backend_zdnn_బఫరు_context * ctx = కొత్త ggml_backend_zdnn_బఫరు_context();

    స్థిరరాశి size_t కొలత_పుట = sysconf(_SC_PAGESIZE);

    size_t కొలత_aligned = size;
    ఒకవేళ ((కొలత_aligned % కొలత_పుట) != 0) {
        కొలత_aligned += కొలత_పుట - (కొలత_aligned % కొలత_పుట);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    పూర్ణాంకం device = ctx_dev->zdnn_device; GGML_వాడని(device);

    ctx->all_దత్తాంశం  = ggml_aligned_malloc(కొలత_aligned);
    ctx->all_కొలత  = కొలత_aligned;
    ctx->owned     = నిజం;
    ctx->n_buffers = 1;

    ఒకవేళ (ctx->all_దత్తాంశం != ఖాళీ) {
        std::unique_ptr<ggml_backend_zdnn_బఫరు> zdnn_బఫరు = std::make_unique<ggml_backend_zdnn_బఫరు>();
        zdnn_బఫరు->దత్తాంశం = ctx->all_దత్తాంశం;
        zdnn_బఫరు->size = కొలత_aligned;
        ctx->buffers.push_back(std::move(zdnn_బఫరు));
    }

    ఒకవేళ (కొలత_aligned > 0 && (ctx->all_దత్తాంశం == ఖాళీ)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, కొలత_aligned / 1024.0 / 1024.0);
        తొలగించు ctx;
        తిరిగి ఖాళీ;
    }

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_zdnn_బఫరు_i, ctx, size);
}

స్థిర size_t ggml_backend_zdnn_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి 256;

    GGML_వాడని(buft);
}

స్థిర తార్కిక ggml_backend_zdnn_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    తిరిగి అబద్ధం;

    GGML_వాడని(buft);
}

ggml_backend_బఫరు_type_t ggml_backend_zdnn_బఫరు_type(శూన్య) {
    స్థిర ggml_backend_బఫరు_type ggml_backend_బఫరు_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_బఫరు_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_బఫరు_type_alloc_బఫరు,
            /* .alloc_buffer_n      = */ ఖాళీ,
            /* .get_alignment       = */ ggml_backend_zdnn_బఫరు_type_get_alignment,
            /* .get_max_size        = */ ఖాళీ,
            /* .get_alloc_size      = */ ఖాళీ,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ఖాళీ,
            /* .is_host             = */ ggml_backend_zdnn_బఫరు_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ఖాళీ,
    };

    తిరిగి &ggml_backend_బఫరు_type_zdnn;
}

//
// backend
//

స్థిర స్థిరరాశి అక్షరం * ggml_backend_zdnn_name(ggml_backend_t backend) {
    తిరిగి GGML_ZDNN_NAME;

    GGML_వాడని(backend);
}

స్థిర శూన్య ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    విడుదల(backend);
}

స్థిర గణన ggml_స్థితి ggml_backend_zdnn_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    తిరిగి ggml_zdnn_లేఖాచిత్రం_compute(backend, cgraph);
}

స్థిర ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ఖాళీ,
    /* .get_tensor_async       = */ ఖాళీ,
    /* .set_tensor_2d_async    = */ ఖాళీ,
    /* .get_tensor_2d_async    = */ ఖాళీ,
    /* .cpy_tensor_async       = */ ఖాళీ,
    /* .synchronize            = */ ఖాళీ,
    /* .graph_plan_create      = */ ఖాళీ,
    /* .graph_plan_free        = */ ఖాళీ,
    /* .graph_plan_update      = */ ఖాళీ,
    /* .graph_plan_compute     = */ ఖాళీ,
    /* .graph_compute          = */ ggml_backend_zdnn_లేఖాచిత్రం_compute,
    /* .event_record           = */ ఖాళీ,
    /* .event_wait             = */ ఖాళీ,
    /* .graph_optimize         = */ ఖాళీ,
};

స్థిర ggml_guid_t ggml_backend_zdnn_guid(శూన్య) {
    స్థిర అక్షరం guid_str[] = "IBM-ZDNN-ACCELER";
    తిరిగి reinterpret_cast<ggml_guid_t>(guid_str);
}

తార్కిక ggml_backend_is_zdnn(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

స్థిర స్థిరరాశి అక్షరం * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    తిరిగి GGML_ZDNN_NAME;

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    తిరిగి "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_zdnn_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    *విడుదల  = 0;
    *మొత్తం = 0;

    GGML_వాడని(dev);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    తిరిగి GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                = */ అబద్ధం,
        /* .host_buffer          = */ అబద్ధం,
        /* .buffer_from_host_ptr = */ అబద్ధం,
        /* .events               = */ అబద్ధం,
        /* .mmap_support         = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    ఒకవేళ (ctx == ఖాళీ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        తిరిగి ఖాళీ;
    }

    ggml_backend_t backend = (ggml_backend *)జ్ఞాపకం_ఇవ్వు(పరిమాణం(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    తిరిగి backend;

    GGML_వాడని(params);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_zdnn_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    తిరిగి ggml_backend_zdnn_బఫరు_type();

    GGML_వాడని(dev);
}

స్థిర తార్కిక ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    తిరిగి ggml_zdnn_supports_op(ctx_dev, op);
}

స్థిర తార్కిక ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    తిరిగి
        buft->iface.get_name == ggml_backend_zdnn_బఫరు_type_get_name;

    GGML_వాడని(dev);
}

స్థిర ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ఖాళీ,
    /* .buffer_from_host_ptr = */ ఖాళీ,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ఖాళీ,
    /* .event_new            = */ ఖాళీ,
    /* .event_free           = */ ఖాళీ,
    /* .event_synchronize    = */ ఖాళీ,
};

//
// backend registry
//

స్థిర స్థిరరాశి అక్షరం * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    తిరిగి GGML_ZDNN_NAME;

    GGML_వాడని(reg);
}

స్థిర size_t ggml_backend_zdnn_reg_device_లెక్క(ggml_backend_reg_t reg) {
    ఒకవేళ (!zdnn_is_nnpa_installed()) {
        తిరిగి 0;
    }
    తిరిగి 1;

    GGML_వాడని(reg);
}

స్థిర ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t సూచీ) {
    GGML_ASSERT(సూచీ == 0);

    తిరిగి &g_ggml_backend_zdnn_device;

    GGML_వాడని(reg);
    GGML_వాడని(సూచీ);
}

స్థిర ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ఖాళీ, ఖాళీ },
};

స్థిర ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    తిరిగి g_ggml_backend_zdnn_features;

    GGML_వాడని(reg);
}

స్థిర శూన్య ggml_backend_zdnn_సమితి_n_దారాలు(ggml_backend_t backend, పూర్ణాంకం n_దారాలు) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_దారాలు = n_దారాలు;
}

స్థిర శూన్య * ggml_backend_zdnn_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (strcmp(name, "ggml_backend_get_features") == 0) {
        తిరిగి (శూన్య *) ggml_backend_zdnn_get_features;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        తిరిగి (శూన్య *) ggml_backend_zdnn_సమితి_n_దారాలు;
    }

    తిరిగి ఖాళీ;

    GGML_వాడని(reg);
}

స్థిర ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_లెక్క,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_చిరునామా
};

స్థిర శూన్య ggml_zdnn_cleanup(శూన్య) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(శూన్య) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_సంస్కరణ,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ ఖాళీ
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        తిరిగి &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
