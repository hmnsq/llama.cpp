#ଅନ୍ତର୍ଭାଵ "ggml-zdnn.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଅନ୍ତର୍ଭାଵ "ggml-zdnn/common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-zdnn/fattn.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-zdnn/mmf.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-zdnn/mmq.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-zdnn/utils.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <csignal>  // raise(SIGTRAP)
#ଅନ୍ତର୍ଭାଵ <unistd.h>

ସ୍ଥିର ଶୂନ୍ୟ ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    ଅଚଳ ggml_tensor * src0 = dst->src[0];  // weights
    ଅଚଳ ggml_tensor * src1 = dst->src[1];  // inputs

    ଯଦି (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } ଅନ୍ୟଥା {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    ବାଛ (dst->op) {
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଗଣନା ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = gf->nodes[i];

        ଯଦି (ggml_is_empty(ଗଣ୍ଠି)
            || ଗଣ୍ଠି->op == GGML_OP_NONE
            || ଗଣ୍ଠି->op == GGML_OP_RESHAPE
            || ଗଣ୍ଠି->op == GGML_OP_VIEW
            || ଗଣ୍ଠି->op == GGML_OP_PERMUTE
            || ଗଣ୍ଠି->op == GGML_OP_TRANSPOSE) {
            ଜାରି;
        }

        ଯଦି ((ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଜାରି;
        }

        ତାର୍କିକ ok = ggml_zdnn_compute_forward(ctx, ଗଣ୍ଠି);
        ଯଦି (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, ଗଣ୍ଠି->name, ggml_op_name(ଗଣ୍ଠି->op));
        }

        GGML_ASSERT(ok);
    }

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

ସ୍ଥିର ତାର୍କିକ ggml_zdnn_supports_op(ଅଚଳ ggml_backend_zdnn_device_context * ctx_dev, ଅଚଳ ggml_tensor * op) {
    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_NONE:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
        ସ୍ଥିତି GGML_OP_PERMUTE:
            ଫେରନ୍ତୁ ସତ୍ୟ;

        ସ୍ଥିତି GGML_OP_MUL_MAT:
            {
                ଅଚଳ ggml_tensor * weights = op->src[0];
                ଅଚଳ ggml_tensor * inputs  = op->src[1];

                ଅଚଳ int64_t ne10 = inputs->ne[0];
                ଅଚଳ int64_t ne0  = op->ne[0];
                ଅଚଳ int64_t ne1  = op->ne[1];

                ଅଚଳ int64_t max_batch = ctx_dev->max_size;

                ଯଦି (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ବାଛ (weights->type) {
                    ସ୍ଥିତି GGML_TYPE_F32:
                    ସ୍ଥିତି GGML_TYPE_F16:
                    ସ୍ଥିତି GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        ଫେରନ୍ତୁ inputs->view_src == nullptr;
                    ସ୍ଥିତି GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        ଫେରନ୍ତୁ ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
            } ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            ଫେରନ୍ତୁ ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
ସ୍ଥିର ggml_backend_reg    g_ggml_backend_zdnn_reg;
ସ୍ଥିର ggml_backend_device g_ggml_backend_zdnn_device;

ସ୍ଥିର ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ ମିଥ୍ୟା,
    /* .has_parmblkformat_1   = */ ମିଥ୍ୟା,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    ପ୍ରତିଜ୍ଞା(ctx != ଖାଲି);

    ଯଦି (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    ଯଦି (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, ଆକାର(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_ଗଣନା++;
    ଫେରନ୍ତୁ ctx->zdnn_device;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    ପ୍ରତିଜ୍ଞା(ctx != ଖାଲି);
    ପ୍ରତିଜ୍ଞା(ctx->zdnn_device_ref_ଗଣନା > 0);

    ctx->zdnn_device_ref_ଗଣନା--;
    ଯଦି (ctx->zdnn_device_ref_ଗଣନା == 0) {
        ଯଦି (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

ସ୍ଥିର ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = ନୂତନ ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    ଫେରନ୍ତୁ ctx;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    ପାଇଁ (ଅଚଳ auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ପାଇଁ (ଅଚଳ auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        ଯଦି (buf->extra != nullptr) ମୁକ୍ତ(buf->extra->data);
        ଯଦି (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    ଯଦି (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    ଫେରନ୍ତୁ ctx->all_data;
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ଯଦି (tensor->view_src != ଖାଲି) {
        ପ୍ରତିଜ୍ଞା(tensor->view_src->buffer->buft == buffer->buft);
        ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    ଯଦି (ggml_is_empty(tensor)) {
        ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ଅଚଳ int64_t tsize = ggml_nbytes(tensor);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    ବାଛ (tensor->op) {
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (ଶୂନ୍ୟ *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ଆକାର(" (bias)"), tensor->name);

                ଅଚଳ int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((ଅକ୍ଷର *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    memcpy((ଅକ୍ଷର *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    ଯଦି (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        ଯଦି (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        ଫେରନ୍ତୁ;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    ଯଦି (extra == nullptr) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        ଫେରନ୍ତୁ;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    ଯଦି (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        ଫେରନ୍ତୁ;
    }
    ଯଦି (extra->ztensor.is_transformed == ମିଥ୍ୟା) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    memcpy(data, (ଅଚଳ ଅକ୍ଷର *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ପାଇଁ (ଅଚଳ auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        ଯଦି (buf->extra != nullptr) {
            ମୁକ୍ତ(buf->extra->data);
        }

        ଯଦି (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    ଯଦି (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

ସ୍ଥିର ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ ଖାଲି,
    /* .get_tensor_2d = */ ଖାଲି,
    /* .cpy_tensor    = */ ଖାଲି,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = ନୂତନ ggml_backend_zdnn_buffer_context();

    ଅଚଳ size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    ଯଦି ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = ସତ୍ୟ;
    ctx->n_buffers = 1;

    ଯଦି (ctx->all_data != ଖାଲି) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    ଯଦି (size_aligned > 0 && (ctx->all_data == ଖାଲି)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

ସ୍ଥିର size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ 256;

    GGML_UNUSED(buft);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    ଫେରନ୍ତୁ ମିଥ୍ୟା;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(ଶୂନ୍ୟ) {
    ସ୍ଥିର ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ଖାଲି,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ ଖାଲି,
            /* .get_alloc_size      = */ ଖାଲି,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ଖାଲି,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ଖାଲି,
    };

    ଫେରନ୍ତୁ &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zdnn_name(ggml_backend_t backend) {
    ଫେରନ୍ତୁ GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    ମୁକ୍ତ(backend);
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ଫେରନ୍ତୁ ggml_zdnn_graph_compute(backend, cgraph);
}

ସ୍ଥିର ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ଖାଲି,
    /* .get_tensor_async       = */ ଖାଲି,
    /* .set_tensor_2d_async    = */ ଖାଲି,
    /* .get_tensor_2d_async    = */ ଖାଲି,
    /* .cpy_tensor_async       = */ ଖାଲି,
    /* .synchronize            = */ ଖାଲି,
    /* .graph_plan_create      = */ ଖାଲି,
    /* .graph_plan_free        = */ ଖାଲି,
    /* .graph_plan_update      = */ ଖାଲି,
    /* .graph_plan_compute     = */ ଖାଲି,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ ଖାଲି,
    /* .event_wait             = */ ଖାଲି,
    /* .graph_optimize         = */ ଖାଲି,
};

ସ୍ଥିର ggml_guid_t ggml_backend_zdnn_guid(ଶୂନ୍ୟ) {
    ସ୍ଥିର ଅକ୍ଷର guid_str[] = "IBM-ZDNN-ACCELER";
    ଫେରନ୍ତୁ reinterpret_cast<ggml_guid_t>(guid_str);
}

ତାର୍କିକ ggml_backend_is_zdnn(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != ଖାଲି &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    *ମୁକ୍ତ  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ ମିଥ୍ୟା,
        /* .host_buffer          = */ ମିଥ୍ୟା,
        /* .buffer_from_host_ptr = */ ମିଥ୍ୟା,
        /* .events               = */ ମିଥ୍ୟା,
        /* .mmap_support         = */ ସତ୍ୟ,
    };
}

ସ୍ଥିର ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    ଯଦି (ctx == ଖାଲି) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(ଆକାର(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    ଫେରନ୍ତୁ backend;

    GGML_UNUSED(params);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    ଫେରନ୍ତୁ ggml_zdnn_supports_op(ctx_dev, op);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ ଖାଲି,
    /* .buffer_from_host_ptr = */ ଖାଲି,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ଖାଲି,
    /* .event_new            = */ ଖାଲି,
    /* .event_free           = */ ଖାଲି,
    /* .event_synchronize    = */ ଖାଲି,
};

//
// backend registry
//

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

ସ୍ଥିର size_t ggml_backend_zdnn_reg_device_ଗଣନା(ggml_backend_reg_t reg) {
    ଯଦି (!zdnn_is_nnpa_installed()) {
        ଫେରନ୍ତୁ 0;
    }
    ଫେରନ୍ତୁ 1;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ଫେରନ୍ତୁ &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

ସ୍ଥିର ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ଖାଲି, ଖାଲି },
};

ସ୍ଥିର ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର * name) {
    ଯଦି (strcmp(name, "ggml_backend_get_features") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ggml_backend_zdnn_get_features;
    }
    ଯଦି (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ggml_backend_zdnn_set_n_threads;
    }

    ଫେରନ୍ତୁ ଖାଲି;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_zdnn_cleanup(ଶୂନ୍ୟ) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(ଶୂନ୍ୟ) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ ଖାଲି
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        ଫେରନ୍ତୁ &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
