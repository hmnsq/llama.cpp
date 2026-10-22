#ಅನ್ತರ್ಭಾವ "ggml-zdnn.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ "ggml-zdnn/common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-zdnn/fattn.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-zdnn/mmf.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-zdnn/mmq.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-zdnn/utils.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <csignal>  // raise(SIGTRAP)
#ಅನ್ತರ್ಭಾವ <unistd.h>

ಸ್ಥಿರ ಶೂನ್ಯ ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];  // weights
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];  // inputs

    ಆದರೆ (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    ಆಯ್ಕೆ (dst->op) {
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_zdnn_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = gf->nodes[i];

        ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(ಗಂಟು)
            || ಗಂಟು->op == GGML_OP_NONE
            || ಗಂಟು->op == GGML_OP_RESHAPE
            || ಗಂಟು->op == GGML_OP_VIEW
            || ಗಂಟು->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ
            || ಗಂಟು->op == GGML_OP_TRANSPOSE) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಮುಂದುವರಿ;
        }

        ತಾರ್ಕಿಕ ok = ggml_zdnn_compute_forward(ctx, ಗಂಟು);
        ಆದರೆ (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, ಗಂಟು->name, ggml_op_name(ಗಂಟು->op));
        }

        GGML_ASSERT(ok);
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;

    GGML_ಬಳಸದ(ctx_dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zdnn_supports_op(ಸ್ಥಿರಾಂಕ ggml_backend_zdnn_device_context * ctx_dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
            ಹಿಂತಿರುಗಿ ನಿಜ;

        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            {
                ಸ್ಥಿರಾಂಕ ggml_tensor * weights = op->src[0];
                ಸ್ಥಿರಾಂಕ ggml_tensor * inputs  = op->src[1];

                ಸ್ಥಿರಾಂಕ int64_t ne10 = inputs->ne[0];
                ಸ್ಥಿರಾಂಕ int64_t ne0  = op->ne[0];
                ಸ್ಥಿರಾಂಕ int64_t ne1  = op->ne[1];

                ಸ್ಥಿರಾಂಕ int64_t max_batch = ctx_dev->max_size;

                ಆದರೆ (!ggml_is_ಮಾತೃಕೆ(weights) || !ggml_is_ಮಾತೃಕೆ(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != ಶೂನ್ಯಸೂಚಕ ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಆಯ್ಕೆ (weights->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        ಹಿಂತಿರುಗಿ inputs->view_src == ಶೂನ್ಯಸೂಚಕ;
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        ಹಿಂತಿರುಗಿ ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            } ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ಹಿಂತಿರುಗಿ ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
ಸ್ಥಿರ ggml_backend_reg    g_ggml_backend_zdnn_reg;
ಸ್ಥಿರ ggml_backend_device g_ggml_backend_zdnn_device;

ಸ್ಥಿರ ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ ಸುಳ್ಳು,
    /* .has_parmblkformat_1   = */ ಸುಳ್ಳು,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    ಪ್ರತಿಜ್ಞಾ(ctx != ಖಾಲಿ);

    ಆದರೆ (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    ಆದರೆ (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_ಗಾತ್ರ();
        strncpy(ctx->name, GGML_ZDNN_NAME, ಗಾತ್ರ(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_ಎಣಿಕೆ++;
    ಹಿಂತಿರುಗಿ ctx->zdnn_device;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    ಪ್ರತಿಜ್ಞಾ(ctx != ಖಾಲಿ);
    ಪ್ರತಿಜ್ಞಾ(ctx->zdnn_device_ref_ಎಣಿಕೆ > 0);

    ctx->zdnn_device_ref_ಎಣಿಕೆ--;
    ಆದರೆ (ctx->zdnn_device_ref_ಎಣಿಕೆ == 0) {
        ಆದರೆ (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

ಸ್ಥಿರ ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ STATIC_LIB
    zdnn_init();
    #ಆದರೆ_ಅಂತ್ಯ

    ggml_backend_zdnn_context * ctx = ಹೊಸ ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    ಪೂರ್ಣಾಂಕ device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_ಎಳೆಗಳು = GGML_DEFAULT_N_ಎಳೆಗಳು;
    ctx->gf = ಶೂನ್ಯಸೂಚಕ;

    ಹಿಂತಿರುಗಿ ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_ಬಫರ್(&it.second->ztensor));
    }

    ಅಳಿಸು ctx;
}

//
// backend interface
//

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_zdnn_ಬಫರ್_context * ctx = (ggml_backend_zdnn_ಬಫರ್_context *)ಬಫರ್->context;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_ಬಫರ್ * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        ಆದರೆ (buf->extra != ಶೂನ್ಯಸೂಚಕ) ಬಿಡುಗಡೆ(buf->extra->ದತ್ತಾಂಶ);
        ಆದರೆ (buf->ztensor.ಬಫರ್_ಗಾತ್ರ > 0) ZDNN_CHECK(zdnn_free_ztensor_ಬಫರ್(&buf->ztensor));
    }

    ಆದರೆ (ctx->owned) {
        ggml_aligned_free(ctx->all_ದತ್ತಾಂಶ, ctx->all_ಗಾತ್ರ);
    }

    ಅಳಿಸು ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_zdnn_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_zdnn_ಬಫರ್_context * ctx = (ggml_backend_zdnn_ಬಫರ್_context *)ಬಫರ್->context;
    ಹಿಂತಿರುಗಿ ctx->all_ದತ್ತಾಂಶ;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_zdnn_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor) {
    ಆದರೆ (tensor->view_src != ಖಾಲಿ) {
        ಪ್ರತಿಜ್ಞಾ(tensor->view_src->ಬಫರ್->buft == ಬಫರ್->buft);
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(tensor)) {
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

    ggml_backend_zdnn_ಬಫರ್_context * ctx = (ggml_backend_zdnn_ಬಫರ್_context *)ಬಫರ್->context;

    ಸ್ಥಿರಾಂಕ int64_t tsize = ggml_nbytes(tensor);
    ಪೂರ್ಣಾಂಕ ಬಫರ್_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_ಬಫರ್> zdnn_ಬಫರ್ = std::make_unique<ggml_backend_zdnn_ಬಫರ್>();
    zdnn_ಬಫರ್->ದತ್ತಾಂಶ = tensor->ದತ್ತಾಂಶ;
    zdnn_ಬಫರ್->size = tsize;
    zdnn_ಬಫರ್->extra = ಶೂನ್ಯಸೂಚಕ;
    snprintf(zdnn_ಬಫರ್->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_ಬಫರ್.get(), tensor);
    tensor->extra = zdnn_ಬಫರ್.get();

    ಆಯ್ಕೆ (tensor->op) {
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_ಬಫರ್> zdnn_bias_ಬಫರ್ = std::make_unique<ggml_backend_zdnn_ಬಫರ್>();
                zdnn_bias_ಬಫರ್->ದತ್ತಾಂಶ = (ಶೂನ್ಯ *)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(tensor->ne[0], ggml_element_ಗಾತ್ರ(tensor));
                zdnn_bias_ಬಫರ್->size = ggml_element_ಗಾತ್ರ(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_ಬಫರ್->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (ಪೂರ್ಣಾಂಕ)ಗಾತ್ರ(" (bias)"), tensor->name);

                ಸ್ಥಿರಾಂಕ int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_ಬಫರ್->pre_tfm_desc,
                                        zdnn_bias_ಬಫರ್->tfm_desc,
                                        zdnn_bias_ಬಫರ್->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_ತುಂಬು_tensor(zdnn_bias_ಬಫರ್->ztensor, zdnn_bias_ಬಫರ್->ದತ್ತಾಂಶ);
                zdnn_ಬಫರ್->extra = zdnn_bias_ಬಫರ್.get();

                ctx->buffers.push_back(std::move(zdnn_bias_ಬಫರ್));
                ctx->n_buffers++;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }

    ctx->buffers.push_back(std::move(zdnn_ಬಫರ್));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;

    GGML_ಬಳಸದ(ಬಫರ್_idx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    memset((ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, value, size);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    ಆದರೆ (tensor->view_src != ಶೂನ್ಯಸೂಚಕ) {
        ggml_backend_zdnn_ಬಫರ್ * src_extra = (ggml_backend_zdnn_ಬಫರ್ *)tensor->view_src->extra;
        ಆದರೆ (src_extra != ಶೂನ್ಯಸೂಚಕ && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_zdnn_ಬಫರ್ * extra = (ggml_backend_zdnn_ಬಫರ್ *)tensor->extra;
    ಆದರೆ (extra == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_ತುಂಬು_tensor_q(extra, tensor);
        ಹಿಂತಿರುಗಿ;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    ಆದರೆ (tensor->ಬಫರ್->usage != GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (extra->ztensor.is_transformed == ಸುಳ್ಳು) ggml_zdnn_ತುಂಬು_tensor(extra->ztensor, tensor->ದತ್ತಾಂಶ);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ggml_backend_zdnn_ಬಫರ್_context * ctx = (ggml_backend_zdnn_ಬಫರ್_context *)ಬಫರ್->context;

    memset(ctx->all_ದತ್ತಾಂಶ, value, ctx->all_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_ಬಫರ್_reset(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_zdnn_ಬಫರ್_context * ctx = (ggml_backend_zdnn_ಬಫರ್_context *)ಬಫರ್->context;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_ಬಫರ್ * buf = buf_ptr.get();

        ಆದರೆ (buf->extra != ಶೂನ್ಯಸೂಚಕ) {
            ಬಿಡುಗಡೆ(buf->extra->ದತ್ತಾಂಶ);
        }

        ಆದರೆ (buf->ztensor.ಬಫರ್_ಗಾತ್ರ > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_ಬಫರ್(&buf->ztensor));
        }
    }

    ಆದರೆ (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_zdnn_ಬಫರ್_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_ಬಫರ್_free_ಬಫರ್,
    /* .get_base      = */ ggml_backend_zdnn_ಬಫರ್_get_ತಳ,
    /* .init_tensor   = */ ggml_backend_zdnn_ಬಫರ್_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_ಬಫರ್_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_ಬಫರ್_get_tensor,
    /* .set_tensor_2d = */ ಖಾಲಿ,
    /* .get_tensor_2d = */ ಖಾಲಿ,
    /* .cpy_tensor    = */ ಖಾಲಿ,
    /* .clear         = */ ggml_backend_zdnn_ಬಫರ್_clear,
    /* .reset         = */ ggml_backend_zdnn_ಬಫರ್_reset,
};

//
// default buffer type
//

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zdnn_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ GGML_ZDNN_NAME;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_zdnn_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ggml_backend_zdnn_ಬಫರ್_context * ctx = ಹೊಸ ggml_backend_zdnn_ಬಫರ್_context();

    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_ಪುಟ = sysconf(_SC_PAGESIZE);

    size_t ಗಾತ್ರ_aligned = size;
    ಆದರೆ ((ಗಾತ್ರ_aligned % ಗಾತ್ರ_ಪುಟ) != 0) {
        ಗಾತ್ರ_aligned += ಗಾತ್ರ_ಪುಟ - (ಗಾತ್ರ_aligned % ಗಾತ್ರ_ಪುಟ);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    ಪೂರ್ಣಾಂಕ device = ctx_dev->zdnn_device; GGML_ಬಳಸದ(device);

    ctx->all_ದತ್ತಾಂಶ  = ggml_aligned_malloc(ಗಾತ್ರ_aligned);
    ctx->all_ಗಾತ್ರ  = ಗಾತ್ರ_aligned;
    ctx->owned     = ನಿಜ;
    ctx->n_buffers = 1;

    ಆದರೆ (ctx->all_ದತ್ತಾಂಶ != ಖಾಲಿ) {
        std::unique_ptr<ggml_backend_zdnn_ಬಫರ್> zdnn_ಬಫರ್ = std::make_unique<ggml_backend_zdnn_ಬಫರ್>();
        zdnn_ಬಫರ್->ದತ್ತಾಂಶ = ctx->all_ದತ್ತಾಂಶ;
        zdnn_ಬಫರ್->size = ಗಾತ್ರ_aligned;
        ctx->buffers.push_back(std::move(zdnn_ಬಫರ್));
    }

    ಆದರೆ (ಗಾತ್ರ_aligned > 0 && (ctx->all_ದತ್ತಾಂಶ == ಖಾಲಿ)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, ಗಾತ್ರ_aligned / 1024.0 / 1024.0);
        ಅಳಿಸು ctx;
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_zdnn_ಬಫರ್_i, ctx, size);
}

ಸ್ಥಿರ size_t ggml_backend_zdnn_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ 256;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zdnn_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(buft);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_zdnn_ಬಫರ್_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_ಬಫರ್_type ggml_backend_ಬಫರ್_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_ಬಫರ್_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_ಬಫರ್_type_alloc_ಬಫರ್,
            /* .alloc_buffer_n      = */ ಖಾಲಿ,
            /* .get_alignment       = */ ggml_backend_zdnn_ಬಫರ್_type_get_alignment,
            /* .get_max_size        = */ ಖಾಲಿ,
            /* .get_alloc_size      = */ ಖಾಲಿ,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ಖಾಲಿ,
            /* .is_host             = */ ggml_backend_zdnn_ಬಫರ್_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ಖಾಲಿ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_ಬಫರ್_type_zdnn;
}

//
// backend
//

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zdnn_name(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ GGML_ZDNN_NAME;

    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    ಬಿಡುಗಡೆ(backend);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_zdnn_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ಹಿಂತಿರುಗಿ ggml_zdnn_ನಕ್ಷೆ_compute(backend, cgraph);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ಖಾಲಿ,
    /* .get_tensor_async       = */ ಖಾಲಿ,
    /* .set_tensor_2d_async    = */ ಖಾಲಿ,
    /* .get_tensor_2d_async    = */ ಖಾಲಿ,
    /* .cpy_tensor_async       = */ ಖಾಲಿ,
    /* .synchronize            = */ ಖಾಲಿ,
    /* .graph_plan_create      = */ ಖಾಲಿ,
    /* .graph_plan_free        = */ ಖಾಲಿ,
    /* .graph_plan_update      = */ ಖಾಲಿ,
    /* .graph_plan_compute     = */ ಖಾಲಿ,
    /* .graph_compute          = */ ggml_backend_zdnn_ನಕ್ಷೆ_compute,
    /* .event_record           = */ ಖಾಲಿ,
    /* .event_wait             = */ ಖಾಲಿ,
    /* .graph_optimize         = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_zdnn_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ಅಕ್ಷರ guid_str[] = "IBM-ZDNN-ACCELER";
    ಹಿಂತಿರುಗಿ reinterpret_cast<ggml_guid_t>(guid_str);
}

ತಾರ್ಕಿಕ ggml_backend_is_zdnn(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_ZDNN_NAME;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    *ಬಿಡುಗಡೆ  = 0;
    *ಒಟ್ಟು = 0;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                = */ ಸುಳ್ಳು,
        /* .host_buffer          = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr = */ ಸುಳ್ಳು,
        /* .events               = */ ಸುಳ್ಳು,
        /* .mmap_support         = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    ಆದರೆ (ctx == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ggml_backend_t backend = (ggml_backend *)ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    ಹಿಂತಿರುಗಿ backend;

    GGML_ಬಳಸದ(params);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_zdnn_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ ggml_backend_zdnn_ಬಫರ್_type();

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    ಹಿಂತಿರುಗಿ ggml_zdnn_supports_op(ctx_dev, op);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ
        buft->iface.get_name == ggml_backend_zdnn_ಬಫರ್_type_get_name;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr = */ ಖಾಲಿ,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ಖಾಲಿ,
    /* .event_new            = */ ಖಾಲಿ,
    /* .event_free           = */ ಖಾಲಿ,
    /* .event_synchronize    = */ ಖಾಲಿ,
};

//
// backend registry
//

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ GGML_ZDNN_NAME;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ size_t ggml_backend_zdnn_reg_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಆದರೆ (!zdnn_is_nnpa_installed()) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಹಿಂತಿರುಗಿ 1;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ಸೂಚ್ಯಂಕ == 0);

    ಹಿಂತಿರುಗಿ &g_ggml_backend_zdnn_device;

    GGML_ಬಳಸದ(reg);
    GGML_ಬಳಸದ(ಸೂಚ್ಯಂಕ);
}

ಸ್ಥಿರ ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ಖಾಲಿ, ಖಾಲಿ },
};

ಸ್ಥಿರ ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ g_ggml_backend_zdnn_features;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_ಗಣ_n_ಎಳೆಗಳು(ggml_backend_t backend, ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_zdnn_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (strcmp(name, "ggml_backend_get_features") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ggml_backend_zdnn_get_features;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ggml_backend_zdnn_ಗಣ_n_ಎಳೆಗಳು;
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_ವಿಳಾಸ
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_zdnn_cleanup(ಶೂನ್ಯ) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(ಶೂನ್ಯ) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_ಆವೃತ್ತಿ,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ ಖಾಲಿ
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        ಹಿಂತಿರುಗಿ &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
