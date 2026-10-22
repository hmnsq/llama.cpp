#അന്തര്ഭാവ "ggml-zdnn.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ "ggml-zdnn/common.hpp"
#അന്തര്ഭാവ "ggml-zdnn/fattn.hpp"
#അന്തര്ഭാവ "ggml-zdnn/mmf.hpp"
#അന്തര്ഭാവ "ggml-zdnn/mmq.hpp"
#അന്തര്ഭാവ "ggml-zdnn/utils.hpp"
#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <csignal>  // raise(SIGTRAP)
#അന്തര്ഭാവ <unistd.h>

സ്ഥിതം ശൂന്യം ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];  // weights
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];  // inputs

    എങ്കിൽ (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } അല്ലെങ്കിൽ {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

സ്ഥിതം തർക്കം ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    തിരഞ്ഞെടുപ്പ് (dst->op) {
        അവസ്ഥ GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } നിർത്തുക;

        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } നിർത്തുക;

        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * കണ്ണി = gf->nodes[i];

        എങ്കിൽ (ggml_is_empty(കണ്ണി)
            || കണ്ണി->op == GGML_OP_NONE
            || കണ്ണി->op == GGML_OP_RESHAPE
            || കണ്ണി->op == GGML_OP_VIEW
            || കണ്ണി->op == GGML_OP_PERMUTE
            || കണ്ണി->op == GGML_OP_TRANSPOSE) {
            തുടരുക;
        }

        എങ്കിൽ ((കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തുടരുക;
        }

        തർക്കം ok = ggml_zdnn_compute_forward(ctx, കണ്ണി);
        എങ്കിൽ (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, കണ്ണി->name, ggml_op_name(കണ്ണി->op));
        }

        GGML_ASSERT(ok);
    }

    തിരികെ GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

സ്ഥിതം തർക്കം ggml_zdnn_supports_op(സ്ഥിരം ggml_backend_zdnn_device_context * ctx_dev, സ്ഥിരം ggml_tensor * op) {
    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_NONE:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_TRANSPOSE:
        അവസ്ഥ GGML_OP_PERMUTE:
            തിരികെ സത്യം;

        അവസ്ഥ GGML_OP_MUL_MAT:
            {
                സ്ഥിരം ggml_tensor * weights = op->src[0];
                സ്ഥിരം ggml_tensor * inputs  = op->src[1];

                സ്ഥിരം int64_t ne10 = inputs->ne[0];
                സ്ഥിരം int64_t ne0  = op->ne[0];
                സ്ഥിരം int64_t ne1  = op->ne[1];

                സ്ഥിരം int64_t max_batch = ctx_dev->max_size;

                എങ്കിൽ (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        തിരികെ അസത്യം;
                }

                തിരഞ്ഞെടുപ്പ് (weights->type) {
                    അവസ്ഥ GGML_TYPE_F32:
                    അവസ്ഥ GGML_TYPE_F16:
                    അവസ്ഥ GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        തിരികെ inputs->view_src == nullptr;
                    അവസ്ഥ GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        തിരികെ ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    സ്ഥിരസ്ഥിതി:
                        തിരികെ അസത്യം;
                }
            } നിർത്തുക;

        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            തിരികെ ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
സ്ഥിതം ggml_backend_reg    g_ggml_backend_zdnn_reg;
സ്ഥിതം ggml_backend_device g_ggml_backend_zdnn_device;

സ്ഥിതം ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ അസത്യം,
    /* .has_parmblkformat_1   = */ അസത്യം,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    പ്രതിജ്ഞാ(ctx != ശൂന്യ);

    എങ്കിൽ (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    എങ്കിൽ (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, വലുപ്പം(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_എണ്ണം++;
    തിരികെ ctx->zdnn_device;
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    പ്രതിജ്ഞാ(ctx != ശൂന്യ);
    പ്രതിജ്ഞാ(ctx->zdnn_device_ref_എണ്ണം > 0);

    ctx->zdnn_device_ref_എണ്ണം--;
    എങ്കിൽ (ctx->zdnn_device_ref_എണ്ണം == 0) {
        എങ്കിൽ (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

സ്ഥിതം ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = പുതിയ ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    പൂർണ്ണസംഖ്യ device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    തിരികെ ctx;
}

സ്ഥിതം ശൂന്യം ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    വേണ്ടി (സ്ഥിരം auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    വേണ്ടി (സ്ഥിരം auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        എങ്കിൽ (buf->extra != nullptr) സ്വതന്ത്രം(buf->extra->data);
        എങ്കിൽ (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    എങ്കിൽ (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

സ്ഥിതം ശൂന്യം * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    തിരികെ ctx->all_data;
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    എങ്കിൽ (tensor->view_src != ശൂന്യ) {
        പ്രതിജ്ഞാ(tensor->view_src->buffer->buft == buffer->buft);
        തിരികെ GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    എങ്കിൽ (ggml_is_empty(tensor)) {
        തിരികെ GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    സ്ഥിരം int64_t tsize = ggml_nbytes(tensor);
    പൂർണ്ണസംഖ്യ buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    തിരഞ്ഞെടുപ്പ് (tensor->op) {
        അവസ്ഥ GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (ശൂന്യം *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (പൂർണ്ണസംഖ്യ)വലുപ്പം(" (bias)"), tensor->name);

                സ്ഥിരം int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    തിരികെ GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((അക്ഷരം *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    memcpy((അക്ഷരം *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    എങ്കിൽ (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        എങ്കിൽ (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        തിരികെ;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    എങ്കിൽ (extra == nullptr) {
        തിരികെ;
    }

    എങ്കിൽ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        തിരികെ;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    എങ്കിൽ (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        തിരികെ;
    }
    എങ്കിൽ (extra->ztensor.is_transformed == അസത്യം) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    memcpy(data, (സ്ഥിരം അക്ഷരം *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    വേണ്ടി (സ്ഥിരം auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        എങ്കിൽ (buf->extra != nullptr) {
            സ്വതന്ത്രം(buf->extra->data);
        }

        എങ്കിൽ (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    എങ്കിൽ (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

സ്ഥിതം ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ ശൂന്യ,
    /* .get_tensor_2d = */ ശൂന്യ,
    /* .cpy_tensor    = */ ശൂന്യ,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    തിരികെ GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = പുതിയ ggml_backend_zdnn_buffer_context();

    സ്ഥിരം size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    എങ്കിൽ ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    പൂർണ്ണസംഖ്യ device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = സത്യം;
    ctx->n_buffers = 1;

    എങ്കിൽ (ctx->all_data != ശൂന്യ) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    എങ്കിൽ (size_aligned > 0 && (ctx->all_data == ശൂന്യ)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        തിരികെ ശൂന്യ;
    }

    തിരികെ ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

സ്ഥിതം size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ 256;

    GGML_UNUSED(buft);
}

സ്ഥിതം തർക്കം ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    തിരികെ അസത്യം;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(ശൂന്യം) {
    സ്ഥിതം ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ശൂന്യ,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ ശൂന്യ,
            /* .get_alloc_size      = */ ശൂന്യ,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ശൂന്യ,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ശൂന്യ,
    };

    തിരികെ &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zdnn_name(ggml_backend_t backend) {
    തിരികെ GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    സ്വതന്ത്രം(backend);
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    തിരികെ ggml_zdnn_graph_compute(backend, cgraph);
}

സ്ഥിതം ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ശൂന്യ,
    /* .get_tensor_async       = */ ശൂന്യ,
    /* .set_tensor_2d_async    = */ ശൂന്യ,
    /* .get_tensor_2d_async    = */ ശൂന്യ,
    /* .cpy_tensor_async       = */ ശൂന്യ,
    /* .synchronize            = */ ശൂന്യ,
    /* .graph_plan_create      = */ ശൂന്യ,
    /* .graph_plan_free        = */ ശൂന്യ,
    /* .graph_plan_update      = */ ശൂന്യ,
    /* .graph_plan_compute     = */ ശൂന്യ,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ ശൂന്യ,
    /* .event_wait             = */ ശൂന്യ,
    /* .graph_optimize         = */ ശൂന്യ,
};

സ്ഥിതം ggml_guid_t ggml_backend_zdnn_guid(ശൂന്യം) {
    സ്ഥിതം അക്ഷരം guid_str[] = "IBM-ZDNN-ACCELER";
    തിരികെ reinterpret_cast<ggml_guid_t>(guid_str);
}

തർക്കം ggml_backend_is_zdnn(ggml_backend_t backend) {
    തിരികെ backend != ശൂന്യ &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    തിരികെ GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    തിരികെ "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    *സ്വതന്ത്രം  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    തിരികെ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ അസത്യം,
        /* .host_buffer          = */ അസത്യം,
        /* .buffer_from_host_ptr = */ അസത്യം,
        /* .events               = */ അസത്യം,
        /* .mmap_support         = */ സത്യം,
    };
}

സ്ഥിതം ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    എങ്കിൽ (ctx == ശൂന്യ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        തിരികെ ശൂന്യ;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(വലുപ്പം(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    തിരികെ backend;

    GGML_UNUSED(params);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    തിരികെ ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

സ്ഥിതം തർക്കം ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    തിരികെ ggml_zdnn_supports_op(ctx_dev, op);
}

സ്ഥിതം തർക്കം ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    തിരികെ
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

സ്ഥിതം ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ ശൂന്യ,
    /* .buffer_from_host_ptr = */ ശൂന്യ,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ശൂന്യ,
    /* .event_new            = */ ശൂന്യ,
    /* .event_free           = */ ശൂന്യ,
    /* .event_synchronize    = */ ശൂന്യ,
};

//
// backend registry
//

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    തിരികെ GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

സ്ഥിതം size_t ggml_backend_zdnn_reg_device_എണ്ണം(ggml_backend_reg_t reg) {
    എങ്കിൽ (!zdnn_is_nnpa_installed()) {
        തിരികെ 0;
    }
    തിരികെ 1;

    GGML_UNUSED(reg);
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    തിരികെ &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

സ്ഥിതം ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ശൂന്യ, ശൂന്യ },
};

സ്ഥിതം ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    തിരികെ g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

സ്ഥിതം ശൂന്യം ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, പൂർണ്ണസംഖ്യ n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

സ്ഥിതം ശൂന്യം * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    എങ്കിൽ (strcmp(name, "ggml_backend_get_features") == 0) {
        തിരികെ (ശൂന്യം *) ggml_backend_zdnn_get_features;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        തിരികെ (ശൂന്യം *) ggml_backend_zdnn_set_n_threads;
    }

    തിരികെ ശൂന്യ;

    GGML_UNUSED(reg);
}

സ്ഥിതം ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

സ്ഥിതം ശൂന്യം ggml_zdnn_cleanup(ശൂന്യം) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(ശൂന്യം) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ ശൂന്യ
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        തിരികെ &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
