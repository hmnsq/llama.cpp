#அந்தர்பாவ "ggml-zdnn.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-backend-impl.h"

#அந்தர்பாவ "ggml-zdnn/common.hpp"
#அந்தர்பாவ "ggml-zdnn/fattn.hpp"
#அந்தர்பாவ "ggml-zdnn/mmf.hpp"
#அந்தர்பாவ "ggml-zdnn/mmq.hpp"
#அந்தர்பாவ "ggml-zdnn/utils.hpp"
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <vector>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <csignal>  // raise(SIGTRAP)
#அந்தர்பாவ <unistd.h>

நிலையான வெறுமை ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];  // weights
    மாறிலி ggml_tensor * src1 = dst->src[1];  // inputs

    எனில் (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } இல்லையெனில் {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

நிலையான தருக்க ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    தேர்வு (dst->op) {
        நிலை GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } நிறுத்து;

        நிலை GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } நிறுத்து;

        இயல்பு:
            திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான எண்ணிக்கை ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    க்கு (முழு i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * முனை = gf->nodes[i];

        எனில் (ggml_is_empty(முனை)
            || முனை->op == GGML_OP_NONE
            || முனை->op == GGML_OP_RESHAPE
            || முனை->op == GGML_OP_VIEW
            || முனை->op == GGML_OP_PERMUTE
            || முனை->op == GGML_OP_TRANSPOSE) {
            தொடர்;
        }

        எனில் ((முனை->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            தொடர்;
        }

        தருக்க ok = ggml_zdnn_compute_forward(ctx, முனை);
        எனில் (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, முனை->name, ggml_op_name(முனை->op));
        }

        GGML_ASSERT(ok);
    }

    திரும்பு GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

நிலையான தருக்க ggml_zdnn_supports_op(மாறிலி ggml_backend_zdnn_device_context * ctx_dev, மாறிலி ggml_tensor * op) {
    தேர்வு (op->op) {
        நிலை GGML_OP_NONE:
        நிலை GGML_OP_RESHAPE:
        நிலை GGML_OP_VIEW:
        நிலை GGML_OP_TRANSPOSE:
        நிலை GGML_OP_PERMUTE:
            திரும்பு உண்மை;

        நிலை GGML_OP_MUL_MAT:
            {
                மாறிலி ggml_tensor * weights = op->src[0];
                மாறிலி ggml_tensor * inputs  = op->src[1];

                மாறிலி int64_t ne10 = inputs->ne[0];
                மாறிலி int64_t ne0  = op->ne[0];
                மாறிலி int64_t ne1  = op->ne[1];

                மாறிலி int64_t max_batch = ctx_dev->max_size;

                எனில் (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        திரும்பு பொய்;
                }

                தேர்வு (weights->type) {
                    நிலை GGML_TYPE_F32:
                    நிலை GGML_TYPE_F16:
                    நிலை GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        திரும்பு inputs->view_src == nullptr;
                    நிலை GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        திரும்பு ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    இயல்பு:
                        திரும்பு பொய்;
                }
            } நிறுத்து;

        நிலை GGML_OP_FLASH_ATTN_EXT:
            திரும்பு ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        இயல்பு:
            திரும்பு பொய்;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
நிலையான ggml_backend_reg    g_ggml_backend_zdnn_reg;
நிலையான ggml_backend_device g_ggml_backend_zdnn_device;

நிலையான ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ பொய்,
    /* .has_parmblkformat_1   = */ பொய்,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

நிலையான முழு ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    ப்ரதிஜ்ஞா(ctx != வெற்று);

    எனில் (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    எனில் (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, அளவு(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_மொத்தம்++;
    திரும்பு ctx->zdnn_device;
}

நிலையான வெறுமை ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    ப்ரதிஜ்ஞா(ctx != வெற்று);
    ப்ரதிஜ்ஞா(ctx->zdnn_device_ref_மொத்தம் > 0);

    ctx->zdnn_device_ref_மொத்தம்--;
    எனில் (ctx->zdnn_device_ref_மொத்தம் == 0) {
        எனில் (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

நிலையான ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = புதிய ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    முழு device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    திரும்பு ctx;
}

நிலையான வெறுமை ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    க்கு (மாறிலி auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

நிலையான வெறுமை ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    க்கு (மாறிலி auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        எனில் (buf->extra != nullptr) விடுவி(buf->extra->data);
        எனில் (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    எனில் (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

நிலையான வெறுமை * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    திரும்பு ctx->all_data;
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    எனில் (tensor->view_src != வெற்று) {
        ப்ரதிஜ்ஞா(tensor->view_src->buffer->buft == buffer->buft);
        திரும்பு GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    எனில் (ggml_is_empty(tensor)) {
        திரும்பு GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    மாறிலி int64_t tsize = ggml_nbytes(tensor);
    முழு buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    தேர்வு (tensor->op) {
        நிலை GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (வெறுமை *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (முழு)அளவு(" (bias)"), tensor->name);

                மாறிலி int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } நிறுத்து;
        இயல்பு:
            நிறுத்து;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    திரும்பு GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

நிலையான வெறுமை ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((எழுத்து *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    memcpy((எழுத்து *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    எனில் (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        எனில் (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        திரும்பு;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    எனில் (extra == nullptr) {
        திரும்பு;
    }

    எனில் (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        திரும்பு;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    எனில் (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        திரும்பு;
    }
    எனில் (extra->ztensor.is_transformed == பொய்) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    memcpy(data, (மாறிலி எழுத்து *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

நிலையான வெறுமை ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    க்கு (மாறிலி auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        எனில் (buf->extra != nullptr) {
            விடுவி(buf->extra->data);
        }

        எனில் (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    எனில் (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

நிலையான ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ வெற்று,
    /* .get_tensor_2d = */ வெற்று,
    /* .cpy_tensor    = */ வெற்று,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

நிலையான மாறிலி எழுத்து * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    திரும்பு GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

நிலையான ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = புதிய ggml_backend_zdnn_buffer_context();

    மாறிலி size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    எனில் ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    முழு device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = உண்மை;
    ctx->n_buffers = 1;

    எனில் (ctx->all_data != வெற்று) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    எனில் (size_aligned > 0 && (ctx->all_data == வெற்று)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        திரும்பு வெற்று;
    }

    திரும்பு ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

நிலையான size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    திரும்பு 256;

    GGML_UNUSED(buft);
}

நிலையான தருக்க ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    திரும்பு பொய்;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(வெறுமை) {
    நிலையான ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ வெற்று,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ வெற்று,
            /* .get_alloc_size      = */ வெற்று,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ வெற்று,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ வெற்று,
    };

    திரும்பு &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

நிலையான மாறிலி எழுத்து * ggml_backend_zdnn_name(ggml_backend_t backend) {
    திரும்பு GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

நிலையான வெறுமை ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    விடுவி(backend);
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    திரும்பு ggml_zdnn_graph_compute(backend, cgraph);
}

நிலையான ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ வெற்று,
    /* .get_tensor_async       = */ வெற்று,
    /* .set_tensor_2d_async    = */ வெற்று,
    /* .get_tensor_2d_async    = */ வெற்று,
    /* .cpy_tensor_async       = */ வெற்று,
    /* .synchronize            = */ வெற்று,
    /* .graph_plan_create      = */ வெற்று,
    /* .graph_plan_free        = */ வெற்று,
    /* .graph_plan_update      = */ வெற்று,
    /* .graph_plan_compute     = */ வெற்று,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ வெற்று,
    /* .event_wait             = */ வெற்று,
    /* .graph_optimize         = */ வெற்று,
};

நிலையான ggml_guid_t ggml_backend_zdnn_guid(வெறுமை) {
    நிலையான எழுத்து guid_str[] = "IBM-ZDNN-ACCELER";
    திரும்பு reinterpret_cast<ggml_guid_t>(guid_str);
}

தருக்க ggml_backend_is_zdnn(ggml_backend_t backend) {
    திரும்பு backend != வெற்று &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

நிலையான மாறிலி எழுத்து * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    திரும்பு GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

நிலையான மாறிலி எழுத்து * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    திரும்பு "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
    *விடுவி  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    திரும்பு GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ பொய்,
        /* .host_buffer          = */ பொய்,
        /* .buffer_from_host_ptr = */ பொய்,
        /* .events               = */ பொய்,
        /* .mmap_support         = */ உண்மை,
    };
}

நிலையான ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    எனில் (ctx == வெற்று) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        திரும்பு வெற்று;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(அளவு(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    திரும்பு backend;

    GGML_UNUSED(params);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    திரும்பு ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

நிலையான தருக்க ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, மாறிலி ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    திரும்பு ggml_zdnn_supports_op(ctx_dev, op);
}

நிலையான தருக்க ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    திரும்பு
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

நிலையான ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ வெற்று,
    /* .buffer_from_host_ptr = */ வெற்று,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ வெற்று,
    /* .event_new            = */ வெற்று,
    /* .event_free           = */ வெற்று,
    /* .event_synchronize    = */ வெற்று,
};

//
// backend registry
//

நிலையான மாறிலி எழுத்து * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    திரும்பு GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

நிலையான size_t ggml_backend_zdnn_reg_device_மொத்தம்(ggml_backend_reg_t reg) {
    எனில் (!zdnn_is_nnpa_installed()) {
        திரும்பு 0;
    }
    திரும்பு 1;

    GGML_UNUSED(reg);
}

நிலையான ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    திரும்பு &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

நிலையான ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { வெற்று, வெற்று },
};

நிலையான ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    திரும்பு g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

நிலையான வெறுமை ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, முழு n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

நிலையான வெறுமை * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    எனில் (strcmp(name, "ggml_backend_get_features") == 0) {
        திரும்பு (வெறுமை *) ggml_backend_zdnn_get_features;
    }
    எனில் (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        திரும்பு (வெறுமை *) ggml_backend_zdnn_set_n_threads;
    }

    திரும்பு வெற்று;

    GGML_UNUSED(reg);
}

நிலையான ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

நிலையான வெறுமை ggml_zdnn_cleanup(வெறுமை) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(வெறுமை) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ வெற்று
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        திரும்பு &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
