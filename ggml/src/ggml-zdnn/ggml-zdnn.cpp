#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zdnn.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zdnn/common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zdnn/fattn.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zdnn/mmf.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zdnn/mmq.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zdnn/utils.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <csignal>  // raise(SIGTRAP)
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

ਸਥਿਰ ਖਾਲੀ ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * src0 = dst->src[0];  // weights
    ਅਚਲ ggml_tensor * src1 = dst->src[1];  // inputs

    ਜੇ (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } ਨਹੀਂਤਾਂ {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    ਚੋਣ (dst->op) {
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } ਰੋਕੋ;

        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_zdnn_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    ਲਈ (ਪੂਰਨ i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = gf->nodes[i];

        ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ)
            || ਗੰਢ->op == GGML_OP_NONE
            || ਗੰਢ->op == GGML_OP_RESHAPE
            || ਗੰਢ->op == GGML_OP_VIEW
            || ਗੰਢ->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ
            || ਗੰਢ->op == GGML_OP_TRANSPOSE) {
            ਜਾਰੀ;
        }

        ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਜਾਰੀ;
        }

        ਤਾਰਕਿਕ ok = ggml_zdnn_compute_forward(ctx, ਗੰਢ);
        ਜੇ (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, ਗੰਢ->name, ggml_op_name(ਗੰਢ->op));
        }

        GGML_ASSERT(ok);
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;

    GGML_ਅਣਵਰਤਿਆ(ctx_dev);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_zdnn_supports_op(ਅਚਲ ggml_backend_zdnn_device_context * ctx_dev, ਅਚਲ ggml_tensor * op) {
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
            ਵਾਪਸ ਸੱਚ;

        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                ਅਚਲ ggml_tensor * weights = op->src[0];
                ਅਚਲ ggml_tensor * inputs  = op->src[1];

                ਅਚਲ int64_t ne10 = inputs->ne[0];
                ਅਚਲ int64_t ne0  = op->ne[0];
                ਅਚਲ int64_t ne1  = op->ne[1];

                ਅਚਲ int64_t max_batch = ctx_dev->max_size;

                ਜੇ (!ggml_is_ਮੈਟ੍ਰਿਕਸ(weights) || !ggml_is_ਮੈਟ੍ਰਿਕਸ(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != ਖਾਲੀ_ਸੂਚਕ ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        ਵਾਪਸ ਝੂਠ;
                }

                ਚੋਣ (weights->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        ਵਾਪਸ inputs->view_src == ਖਾਲੀ_ਸੂਚਕ;
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        ਵਾਪਸ ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            } ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ਵਾਪਸ ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
ਸਥਿਰ ggml_backend_reg    g_ggml_backend_zdnn_reg;
ਸਥਿਰ ggml_backend_device g_ggml_backend_zdnn_device;

ਸਥਿਰ ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ ਝੂਠ,
    /* .has_parmblkformat_1   = */ ਝੂਠ,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

ਸਥਿਰ ਪੂਰਨ ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    ਪ੍ਰਤਿਜ੍ਞਾ(ctx != ਨਿਰਾ);

    ਜੇ (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    ਜੇ (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_ਮਾਪ();
        strncpy(ctx->name, GGML_ZDNN_NAME, ਆਕਾਰ(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_ਗਿਣਤੀ++;
    ਵਾਪਸ ctx->zdnn_device;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    ਪ੍ਰਤਿਜ੍ਞਾ(ctx != ਨਿਰਾ);
    ਪ੍ਰਤਿਜ੍ਞਾ(ctx->zdnn_device_ref_ਗਿਣਤੀ > 0);

    ctx->zdnn_device_ref_ਗਿਣਤੀ--;
    ਜੇ (ctx->zdnn_device_ref_ਗਿਣਤੀ == 0) {
        ਜੇ (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

ਸਥਿਰ ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ STATIC_LIB
    zdnn_init();
    #ਜੇ_ਅੰਤ

    ggml_backend_zdnn_context * ctx = ਨਵਾਂ ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    ਪੂਰਨ device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_ਧਾਗੇ = GGML_DEFAULT_N_ਧਾਗੇ;
    ctx->gf = ਖਾਲੀ_ਸੂਚਕ;

    ਵਾਪਸ ctx;
}

ਸਥਿਰ ਖਾਲੀ ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_ਬਫਰ(&it.second->ztensor));
    }

    ਮਿਟਾਓ ctx;
}

//
// backend interface
//

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_zdnn_ਬਫਰ_context * ctx = (ggml_backend_zdnn_ਬਫਰ_context *)ਬਫਰ->context;

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_ਬਫਰ * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        ਜੇ (buf->extra != ਖਾਲੀ_ਸੂਚਕ) ਮੁਕਤ(buf->extra->ਅੰਕੜੇ);
        ਜੇ (buf->ztensor.ਬਫਰ_ਮਾਪ > 0) ZDNN_CHECK(zdnn_free_ztensor_ਬਫਰ(&buf->ztensor));
    }

    ਜੇ (ctx->owned) {
        ggml_aligned_free(ctx->all_ਅੰਕੜੇ, ctx->all_ਮਾਪ);
    }

    ਮਿਟਾਓ ctx;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_zdnn_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_zdnn_ਬਫਰ_context * ctx = (ggml_backend_zdnn_ਬਫਰ_context *)ਬਫਰ->context;
    ਵਾਪਸ ctx->all_ਅੰਕੜੇ;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_zdnn_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor) {
    ਜੇ (tensor->view_src != ਨਿਰਾ) {
        ਪ੍ਰਤਿਜ੍ਞਾ(tensor->view_src->ਬਫਰ->buft == ਬਫਰ->buft);
        ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(tensor)) {
        ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
    }

    ggml_backend_zdnn_ਬਫਰ_context * ctx = (ggml_backend_zdnn_ਬਫਰ_context *)ਬਫਰ->context;

    ਅਚਲ int64_t tsize = ggml_nbytes(tensor);
    ਪੂਰਨ ਬਫਰ_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_ਬਫਰ> zdnn_ਬਫਰ = std::make_unique<ggml_backend_zdnn_ਬਫਰ>();
    zdnn_ਬਫਰ->ਅੰਕੜੇ = tensor->ਅੰਕੜੇ;
    zdnn_ਬਫਰ->size = tsize;
    zdnn_ਬਫਰ->extra = ਖਾਲੀ_ਸੂਚਕ;
    snprintf(zdnn_ਬਫਰ->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_ਬਫਰ.get(), tensor);
    tensor->extra = zdnn_ਬਫਰ.get();

    ਚੋਣ (tensor->op) {
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_ਬਫਰ> zdnn_bias_ਬਫਰ = std::make_unique<ggml_backend_zdnn_ਬਫਰ>();
                zdnn_bias_ਬਫਰ->ਅੰਕੜੇ = (ਖਾਲੀ *)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(tensor->ne[0], ggml_element_ਮਾਪ(tensor));
                zdnn_bias_ਬਫਰ->size = ggml_element_ਮਾਪ(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_ਬਫਰ->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (ਪੂਰਨ)ਆਕਾਰ(" (bias)"), tensor->name);

                ਅਚਲ int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_ਬਫਰ->pre_tfm_desc,
                                        zdnn_bias_ਬਫਰ->tfm_desc,
                                        zdnn_bias_ਬਫਰ->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_ਲੱਦੋ_tensor(zdnn_bias_ਬਫਰ->ztensor, zdnn_bias_ਬਫਰ->ਅੰਕੜੇ);
                zdnn_ਬਫਰ->extra = zdnn_bias_ਬਫਰ.get();

                ctx->buffers.push_back(std::move(zdnn_bias_ਬਫਰ));
                ctx->n_buffers++;
            } ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }

    ctx->buffers.push_back(std::move(zdnn_ਬਫਰ));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ_idx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    memset((ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, value, size);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    ਜੇ (tensor->view_src != ਖਾਲੀ_ਸੂਚਕ) {
        ggml_backend_zdnn_ਬਫਰ * src_extra = (ggml_backend_zdnn_ਬਫਰ *)tensor->view_src->extra;
        ਜੇ (src_extra != ਖਾਲੀ_ਸੂਚਕ && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        ਵਾਪਸ;
    }

    ggml_backend_zdnn_ਬਫਰ * extra = (ggml_backend_zdnn_ਬਫਰ *)tensor->extra;
    ਜੇ (extra == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    ਜੇ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ਵਿੱਥ == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_ਲੱਦੋ_tensor_q(extra, tensor);
        ਵਾਪਸ;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    ਜੇ (tensor->ਬਫਰ->usage != GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        ਵਾਪਸ;
    }
    ਜੇ (extra->ztensor.is_transformed == ਝੂਠ) ggml_zdnn_ਲੱਦੋ_tensor(extra->ztensor, tensor->ਅੰਕੜੇ);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, size);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ggml_backend_zdnn_ਬਫਰ_context * ctx = (ggml_backend_zdnn_ਬਫਰ_context *)ਬਫਰ->context;

    memset(ctx->all_ਅੰਕੜੇ, value, ctx->all_ਮਾਪ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_ਬਫਰ_reset(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_zdnn_ਬਫਰ_context * ctx = (ggml_backend_zdnn_ਬਫਰ_context *)ਬਫਰ->context;

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_ਬਫਰ * buf = buf_ptr.get();

        ਜੇ (buf->extra != ਖਾਲੀ_ਸੂਚਕ) {
            ਮੁਕਤ(buf->extra->ਅੰਕੜੇ);
        }

        ਜੇ (buf->ztensor.ਬਫਰ_ਮਾਪ > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_ਬਫਰ(&buf->ztensor));
        }
    }

    ਜੇ (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_zdnn_ਬਫਰ_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_ਬਫਰ_free_ਬਫਰ,
    /* .get_base      = */ ggml_backend_zdnn_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor   = */ ggml_backend_zdnn_ਬਫਰ_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_ਬਫਰ_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_ਬਫਰ_get_tensor,
    /* .set_tensor_2d = */ ਨਿਰਾ,
    /* .get_tensor_2d = */ ਨਿਰਾ,
    /* .cpy_tensor    = */ ਨਿਰਾ,
    /* .clear         = */ ggml_backend_zdnn_ਬਫਰ_clear,
    /* .reset         = */ ggml_backend_zdnn_ਬਫਰ_reset,
};

//
// default buffer type
//

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zdnn_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ GGML_ZDNN_NAME;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_zdnn_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ggml_backend_zdnn_ਬਫਰ_context * ctx = ਨਵਾਂ ggml_backend_zdnn_ਬਫਰ_context();

    ਅਚਲ size_t ਮਾਪ_ਪੰਨਾ = sysconf(_SC_PAGESIZE);

    size_t ਮਾਪ_aligned = size;
    ਜੇ ((ਮਾਪ_aligned % ਮਾਪ_ਪੰਨਾ) != 0) {
        ਮਾਪ_aligned += ਮਾਪ_ਪੰਨਾ - (ਮਾਪ_aligned % ਮਾਪ_ਪੰਨਾ);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    ਪੂਰਨ device = ctx_dev->zdnn_device; GGML_ਅਣਵਰਤਿਆ(device);

    ctx->all_ਅੰਕੜੇ  = ggml_aligned_malloc(ਮਾਪ_aligned);
    ctx->all_ਮਾਪ  = ਮਾਪ_aligned;
    ctx->owned     = ਸੱਚ;
    ctx->n_buffers = 1;

    ਜੇ (ctx->all_ਅੰਕੜੇ != ਨਿਰਾ) {
        std::unique_ptr<ggml_backend_zdnn_ਬਫਰ> zdnn_ਬਫਰ = std::make_unique<ggml_backend_zdnn_ਬਫਰ>();
        zdnn_ਬਫਰ->ਅੰਕੜੇ = ctx->all_ਅੰਕੜੇ;
        zdnn_ਬਫਰ->size = ਮਾਪ_aligned;
        ctx->buffers.push_back(std::move(zdnn_ਬਫਰ));
    }

    ਜੇ (ਮਾਪ_aligned > 0 && (ctx->all_ਅੰਕੜੇ == ਨਿਰਾ)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, ਮਾਪ_aligned / 1024.0 / 1024.0);
        ਮਿਟਾਓ ctx;
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_zdnn_ਬਫਰ_i, ctx, size);
}

ਸਥਿਰ size_t ggml_backend_zdnn_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ 256;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_zdnn_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_zdnn_ਬਫਰ_type(ਖਾਲੀ) {
    ਸਥਿਰ ggml_backend_ਬਫਰ_type ggml_backend_ਬਫਰ_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_ਬਫਰ_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_ਬਫਰ_type_alloc_ਬਫਰ,
            /* .alloc_buffer_n      = */ ਨਿਰਾ,
            /* .get_alignment       = */ ggml_backend_zdnn_ਬਫਰ_type_get_alignment,
            /* .get_max_size        = */ ਨਿਰਾ,
            /* .get_alloc_size      = */ ਨਿਰਾ,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ਨਿਰਾ,
            /* .is_host             = */ ggml_backend_zdnn_ਬਫਰ_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ਨਿਰਾ,
    };

    ਵਾਪਸ &ggml_backend_ਬਫਰ_type_zdnn;
}

//
// backend
//

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zdnn_name(ggml_backend_t backend) {
    ਵਾਪਸ GGML_ZDNN_NAME;

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    ਮੁਕਤ(backend);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_zdnn_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ਵਾਪਸ ggml_zdnn_ਗ੍ਰਾਫ_compute(backend, cgraph);
}

ਸਥਿਰ ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ਨਿਰਾ,
    /* .get_tensor_async       = */ ਨਿਰਾ,
    /* .set_tensor_2d_async    = */ ਨਿਰਾ,
    /* .get_tensor_2d_async    = */ ਨਿਰਾ,
    /* .cpy_tensor_async       = */ ਨਿਰਾ,
    /* .synchronize            = */ ਨਿਰਾ,
    /* .graph_plan_create      = */ ਨਿਰਾ,
    /* .graph_plan_free        = */ ਨਿਰਾ,
    /* .graph_plan_update      = */ ਨਿਰਾ,
    /* .graph_plan_compute     = */ ਨਿਰਾ,
    /* .graph_compute          = */ ggml_backend_zdnn_ਗ੍ਰਾਫ_compute,
    /* .event_record           = */ ਨਿਰਾ,
    /* .event_wait             = */ ਨਿਰਾ,
    /* .graph_optimize         = */ ਨਿਰਾ,
};

ਸਥਿਰ ggml_guid_t ggml_backend_zdnn_guid(ਖਾਲੀ) {
    ਸਥਿਰ ਅੱਖਰ guid_str[] = "IBM-ZDNN-ACCELER";
    ਵਾਪਸ reinterpret_cast<ggml_guid_t>(guid_str);
}

ਤਾਰਕਿਕ ggml_backend_is_zdnn(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_ZDNN_NAME;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    ਵਾਪਸ "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    *ਮੁਕਤ  = 0;
    *ਕੁੱਲ = 0;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                = */ ਝੂਠ,
        /* .host_buffer          = */ ਝੂਠ,
        /* .buffer_from_host_ptr = */ ਝੂਠ,
        /* .events               = */ ਝੂਠ,
        /* .mmap_support         = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    ਜੇ (ctx == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ggml_backend_t backend = (ggml_backend *)ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    ਵਾਪਸ backend;

    GGML_ਅਣਵਰਤਿਆ(params);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_zdnn_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ ggml_backend_zdnn_ਬਫਰ_type();

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    ਵਾਪਸ ggml_zdnn_supports_op(ctx_dev, op);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ
        buft->iface.get_name == ggml_backend_zdnn_ਬਫਰ_type_get_name;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr = */ ਨਿਰਾ,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ਨਿਰਾ,
    /* .event_new            = */ ਨਿਰਾ,
    /* .event_free           = */ ਨਿਰਾ,
    /* .event_synchronize    = */ ਨਿਰਾ,
};

//
// backend registry
//

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    ਵਾਪਸ GGML_ZDNN_NAME;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ size_t ggml_backend_zdnn_reg_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ਜੇ (!zdnn_is_nnpa_installed()) {
        ਵਾਪਸ 0;
    }
    ਵਾਪਸ 1;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ਸੂਚਕਾਂਕ == 0);

    ਵਾਪਸ &g_ggml_backend_zdnn_device;

    GGML_ਅਣਵਰਤਿਆ(reg);
    GGML_ਅਣਵਰਤਿਆ(ਸੂਚਕਾਂਕ);
}

ਸਥਿਰ ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ਨਿਰਾ, ਨਿਰਾ },
};

ਸਥਿਰ ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    ਵਾਪਸ g_ggml_backend_zdnn_features;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_zdnn_ਸੈੱਟ_n_ਧਾਗੇ(ggml_backend_t backend, ਪੂਰਨ n_ਧਾਗੇ) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_ਧਾਗੇ = n_ਧਾਗੇ;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_zdnn_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (strcmp(name, "ggml_backend_get_features") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *) ggml_backend_zdnn_get_features;
    }
    ਜੇ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *) ggml_backend_zdnn_ਸੈੱਟ_n_ਧਾਗੇ;
    }

    ਵਾਪਸ ਨਿਰਾ;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_ਪਤਾ
};

ਸਥਿਰ ਖਾਲੀ ggml_zdnn_cleanup(ਖਾਲੀ) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(ਖਾਲੀ) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_ਸੰਸਕਰਣ,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ ਨਿਰਾ
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        ਵਾਪਸ &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
