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

    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];  // weights
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];  // inputs

    ಯದಿ (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } ಅನ್ಯಥಾ {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    ವಿಕಲ್ಪ (dst->op) {
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } ಭಙ್ಗ;

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = gf->nodes[i];

        ಯದಿ (ggml_is_empty(ಗಂಟು)
            || ಗಂಟು->op == GGML_OP_NONE
            || ಗಂಟು->op == GGML_OP_RESHAPE
            || ಗಂಟು->op == GGML_OP_VIEW
            || ಗಂಟು->op == GGML_OP_PERMUTE
            || ಗಂಟು->op == GGML_OP_TRANSPOSE) {
            ಅನುವರ್ತ;
        }

        ಯದಿ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಅನುವರ್ತ;
        }

        ತಾರ್ಕಿಕ ok = ggml_zdnn_compute_forward(ctx, ಗಂಟು);
        ಯದಿ (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, ಗಂಟು->name, ggml_op_name(ಗಂಟು->op));
        }

        GGML_ASSERT(ok);
    }

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_zdnn_supports_op(ಧ್ರುವ ggml_backend_zdnn_device_context * ctx_dev, ಧ್ರುವ ggml_tensor * op) {
    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_NONE:
        ಪಕ್ಷ GGML_OP_RESHAPE:
        ಪಕ್ಷ GGML_OP_VIEW:
        ಪಕ್ಷ GGML_OP_TRANSPOSE:
        ಪಕ್ಷ GGML_OP_PERMUTE:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

        ಪಕ್ಷ GGML_OP_MUL_MAT:
            {
                ಧ್ರುವ ggml_tensor * weights = op->src[0];
                ಧ್ರುವ ggml_tensor * inputs  = op->src[1];

                ಧ್ರುವ int64_t ne10 = inputs->ne[0];
                ಧ್ರುವ int64_t ne0  = op->ne[0];
                ಧ್ರುವ int64_t ne1  = op->ne[1];

                ಧ್ರುವ int64_t max_batch = ctx_dev->max_size;

                ಯದಿ (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }

                ವಿಕಲ್ಪ (weights->type) {
                    ಪಕ್ಷ GGML_TYPE_F32:
                    ಪಕ್ಷ GGML_TYPE_F16:
                    ಪಕ್ಷ GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        ಪ್ರತ್ಯಾಗಮ inputs->view_src == nullptr;
                    ಪಕ್ಷ GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        ಪ್ರತ್ಯಾಗಮ ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    ಸ್ವಭಾವ:
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            } ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            ಪ್ರತ್ಯಾಗಮ ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
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
    /* .has_parmblkformat_0   = */ ಅಸತ್ಯ,
    /* .has_parmblkformat_1   = */ ಅಸತ್ಯ,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    ಪ್ರತಿಜ್ಞಾ(ctx != ರಿಕ್ತ);

    ಯದಿ (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    ಯದಿ (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, ಪರಿಮಾಣ(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_ಎಣಿಕೆ++;
    ಪ್ರತ್ಯಾಗಮ ctx->zdnn_device;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    ಪ್ರತಿಜ್ಞಾ(ctx != ರಿಕ್ತ);
    ಪ್ರತಿಜ್ಞಾ(ctx->zdnn_device_ref_ಎಣಿಕೆ > 0);

    ctx->zdnn_device_ref_ಎಣಿಕೆ--;
    ಯದಿ (ctx->zdnn_device_ref_ಎಣಿಕೆ == 0) {
        ಯದಿ (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

ಸ್ಥಿರ ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = ನೂತನ ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    ಪೂರ್ಣಾಙ್ಕ device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    ಪ್ರತ್ಯಾಗಮ ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    ಪ್ರತಿ (ಧ್ರುವ auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ಪ್ರತಿ (ಧ್ರುವ auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        ಯದಿ (buf->extra != nullptr) ಮುಕ್ತ(buf->extra->data);
        ಯದಿ (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    ಯದಿ (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    ಪ್ರತ್ಯಾಗಮ ctx->all_data;
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ಯದಿ (tensor->view_src != ರಿಕ್ತ) {
        ಪ್ರತಿಜ್ಞಾ(tensor->view_src->buffer->buft == buffer->buft);
        ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    ಯದಿ (ggml_is_empty(tensor)) {
        ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ಧ್ರುವ int64_t tsize = ggml_nbytes(tensor);
    ಪೂರ್ಣಾಙ್ಕ buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    ವಿಕಲ್ಪ (tensor->op) {
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (ಶೂನ್ಯ *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (ಪೂರ್ಣಾಙ್ಕ)ಪರಿಮಾಣ(" (bias)"), tensor->name);

                ಧ್ರುವ int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((ವರ್ಣ *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    memcpy((ವರ್ಣ *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    ಯದಿ (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        ಯದಿ (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    ಯದಿ (extra == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        ಪ್ರತ್ಯಾಗಮ;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    ಯದಿ (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (extra->ztensor.is_transformed == ಅಸತ್ಯ) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    memcpy(data, (ಧ್ರುವ ವರ್ಣ *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ಪ್ರತಿ (ಧ್ರುವ auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        ಯದಿ (buf->extra != nullptr) {
            ಮುಕ್ತ(buf->extra->data);
        }

        ಯದಿ (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    ಯದಿ (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

ಸ್ಥಿರ ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ ರಿಕ್ತ,
    /* .get_tensor_2d = */ ರಿಕ್ತ,
    /* .cpy_tensor    = */ ರಿಕ್ತ,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = ನೂತನ ggml_backend_zdnn_buffer_context();

    ಧ್ರುವ size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    ಯದಿ ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    ಪೂರ್ಣಾಙ್ಕ device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = ಸತ್ಯ;
    ctx->n_buffers = 1;

    ಯದಿ (ctx->all_data != ರಿಕ್ತ) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    ಯದಿ (size_aligned > 0 && (ctx->all_data == ರಿಕ್ತ)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

ಸ್ಥಿರ size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ 256;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ರಿಕ್ತ,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ ರಿಕ್ತ,
            /* .get_alloc_size      = */ ರಿಕ್ತ,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ರಿಕ್ತ,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ರಿಕ್ತ,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_zdnn_name(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    ಮುಕ್ತ(backend);
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ಪ್ರತ್ಯಾಗಮ ggml_zdnn_graph_compute(backend, cgraph);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ರಿಕ್ತ,
    /* .get_tensor_async       = */ ರಿಕ್ತ,
    /* .set_tensor_2d_async    = */ ರಿಕ್ತ,
    /* .get_tensor_2d_async    = */ ರಿಕ್ತ,
    /* .cpy_tensor_async       = */ ರಿಕ್ತ,
    /* .synchronize            = */ ರಿಕ್ತ,
    /* .graph_plan_create      = */ ರಿಕ್ತ,
    /* .graph_plan_free        = */ ರಿಕ್ತ,
    /* .graph_plan_update      = */ ರಿಕ್ತ,
    /* .graph_plan_compute     = */ ರಿಕ್ತ,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ ರಿಕ್ತ,
    /* .event_wait             = */ ರಿಕ್ತ,
    /* .graph_optimize         = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_zdnn_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ವರ್ಣ guid_str[] = "IBM-ZDNN-ACCELER";
    ಪ್ರತ್ಯಾಗಮ reinterpret_cast<ggml_guid_t>(guid_str);
}

ತಾರ್ಕಿಕ ggml_backend_is_zdnn(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend != ರಿಕ್ತ &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
    *ಮುಕ್ತ  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ ಅಸತ್ಯ,
        /* .host_buffer          = */ ಅಸತ್ಯ,
        /* .buffer_from_host_ptr = */ ಅಸತ್ಯ,
        /* .events               = */ ಅಸತ್ಯ,
        /* .mmap_support         = */ ಸತ್ಯ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    ಯದಿ (ctx == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(ಪರಿಮಾಣ(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    ಪ್ರತ್ಯಾಗಮ backend;

    GGML_UNUSED(params);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    ಪ್ರತ್ಯಾಗಮ ggml_zdnn_supports_op(ctx_dev, op);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ ರಿಕ್ತ,
    /* .buffer_from_host_ptr = */ ರಿಕ್ತ,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ರಿಕ್ತ,
    /* .event_new            = */ ರಿಕ್ತ,
    /* .event_free           = */ ರಿಕ್ತ,
    /* .event_synchronize    = */ ರಿಕ್ತ,
};

//
// backend registry
//

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ size_t ggml_backend_zdnn_reg_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಯದಿ (!zdnn_is_nnpa_installed()) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಪ್ರತ್ಯಾಗಮ 1;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ಪ್ರತ್ಯಾಗಮ &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

ಸ್ಥಿರ ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ರಿಕ್ತ, ರಿಕ್ತ },
};

ಸ್ಥಿರ ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, ಪೂರ್ಣಾಙ್ಕ n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, ಧ್ರುವ ವರ್ಣ * name) {
    ಯದಿ (strcmp(name, "ggml_backend_get_features") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) ggml_backend_zdnn_get_features;
    }
    ಯದಿ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) ggml_backend_zdnn_set_n_threads;
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
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
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ ರಿಕ್ತ
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        ಪ್ರತ್ಯಾಗಮ &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
