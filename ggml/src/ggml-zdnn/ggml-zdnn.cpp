#અન્તર્ભાવ "ggml-zdnn.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-backend-impl.h"

#અન્તર્ભાવ "ggml-zdnn/common.hpp"
#અન્તર્ભાવ "ggml-zdnn/fattn.hpp"
#અન્તર્ભાવ "ggml-zdnn/mmf.hpp"
#અન્તર્ભાવ "ggml-zdnn/mmq.hpp"
#અન્તર્ભાવ "ggml-zdnn/utils.hpp"
#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <csignal>  // raise(SIGTRAP)
#અન્તર્ભાવ <unistd.h>

સ્થિર શૂન્ય ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    અચલ ggml_tensor * src0 = dst->src[0];  // weights
    અચલ ggml_tensor * src1 = dst->src[1];  // inputs

    જો (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } નહીંતર {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

સ્થિર તાર્કિક ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    પસંદગી (dst->op) {
        સ્થિતિ GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } તોડો;

        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } તોડો;

        મૂળભૂત:
            પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર ગણના ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    માટે (પૂર્ણાંક i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * ગાંઠ = gf->nodes[i];

        જો (ggml_is_empty(ગાંઠ)
            || ગાંઠ->op == GGML_OP_NONE
            || ગાંઠ->op == GGML_OP_RESHAPE
            || ગાંઠ->op == GGML_OP_VIEW
            || ગાંઠ->op == GGML_OP_PERMUTE
            || ગાંઠ->op == GGML_OP_TRANSPOSE) {
            ચાલુ;
        }

        જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ચાલુ;
        }

        તાર્કિક ok = ggml_zdnn_compute_forward(ctx, ગાંઠ);
        જો (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, ગાંઠ->name, ggml_op_name(ગાંઠ->op));
        }

        GGML_ASSERT(ok);
    }

    પાછા GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

સ્થિર તાર્કિક ggml_zdnn_supports_op(અચલ ggml_backend_zdnn_device_context * ctx_dev, અચલ ggml_tensor * op) {
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_TRANSPOSE:
        સ્થિતિ GGML_OP_PERMUTE:
            પાછા સાચું;

        સ્થિતિ GGML_OP_MUL_MAT:
            {
                અચલ ggml_tensor * weights = op->src[0];
                અચલ ggml_tensor * inputs  = op->src[1];

                અચલ int64_t ne10 = inputs->ne[0];
                અચલ int64_t ne0  = op->ne[0];
                અચલ int64_t ne1  = op->ne[1];

                અચલ int64_t max_batch = ctx_dev->max_size;

                જો (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        પાછા ખોટું;
                }

                પસંદગી (weights->type) {
                    સ્થિતિ GGML_TYPE_F32:
                    સ્થિતિ GGML_TYPE_F16:
                    સ્થિતિ GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        પાછા inputs->view_src == nullptr;
                    સ્થિતિ GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        પાછા ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    મૂળભૂત:
                        પાછા ખોટું;
                }
            } તોડો;

        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            પાછા ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        મૂળભૂત:
            પાછા ખોટું;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
સ્થિર ggml_backend_reg    g_ggml_backend_zdnn_reg;
સ્થિર ggml_backend_device g_ggml_backend_zdnn_device;

સ્થિર ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ ખોટું,
    /* .has_parmblkformat_1   = */ ખોટું,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

સ્થિર પૂર્ણાંક ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    પ્રતિજ્ઞા(ctx != ખાલી);

    જો (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    જો (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, કદ(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_ગણતરી++;
    પાછા ctx->zdnn_device;
}

સ્થિર શૂન્ય ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    પ્રતિજ્ઞા(ctx != ખાલી);
    પ્રતિજ્ઞા(ctx->zdnn_device_ref_ગણતરી > 0);

    ctx->zdnn_device_ref_ગણતરી--;
    જો (ctx->zdnn_device_ref_ગણતરી == 0) {
        જો (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

સ્થિર ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = નવું ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    પૂર્ણાંક device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    પાછા ctx;
}

સ્થિર શૂન્ય ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    માટે (અચલ auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

સ્થિર શૂન્ય ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    માટે (અચલ auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        જો (buf->extra != nullptr) મુક્ત(buf->extra->data);
        જો (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    જો (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

સ્થિર શૂન્ય * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    પાછા ctx->all_data;
}

સ્થિર ગણના ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    જો (tensor->view_src != ખાલી) {
        પ્રતિજ્ઞા(tensor->view_src->buffer->buft == buffer->buft);
        પાછા GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    જો (ggml_is_empty(tensor)) {
        પાછા GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    અચલ int64_t tsize = ggml_nbytes(tensor);
    પૂર્ણાંક buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    પસંદગી (tensor->op) {
        સ્થિતિ GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (શૂન્ય *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (પૂર્ણાંક)કદ(" (bias)"), tensor->name);

                અચલ int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } તોડો;
        મૂળભૂત:
            તોડો;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    પાછા GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

સ્થિર શૂન્ય ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((અક્ષર *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    memcpy((અક્ષર *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    જો (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        જો (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        પાછા;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    જો (extra == nullptr) {
        પાછા;
    }

    જો (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        પાછા;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    જો (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        પાછા;
    }
    જો (extra->ztensor.is_transformed == ખોટું) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, અચલ ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    memcpy(data, (અચલ અક્ષર *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

સ્થિર શૂન્ય ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    માટે (અચલ auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        જો (buf->extra != nullptr) {
            મુક્ત(buf->extra->data);
        }

        જો (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    જો (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

સ્થિર ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ ખાલી,
    /* .get_tensor_2d = */ ખાલી,
    /* .cpy_tensor    = */ ખાલી,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

સ્થિર અચલ અક્ષર * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    પાછા GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = નવું ggml_backend_zdnn_buffer_context();

    અચલ size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    જો ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    પૂર્ણાંક device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = સાચું;
    ctx->n_buffers = 1;

    જો (ctx->all_data != ખાલી) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    જો (size_aligned > 0 && (ctx->all_data == ખાલી)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        પાછા ખાલી;
    }

    પાછા ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

સ્થિર size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા 256;

    GGML_UNUSED(buft);
}

સ્થિર તાર્કિક ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    પાછા ખોટું;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(શૂન્ય) {
    સ્થિર ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ ખાલી,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ ખાલી,
            /* .get_alloc_size      = */ ખાલી,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ખાલી,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ખાલી,
    };

    પાછા &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

સ્થિર અચલ અક્ષર * ggml_backend_zdnn_name(ggml_backend_t backend) {
    પાછા GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

સ્થિર શૂન્ય ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    મુક્ત(backend);
}

સ્થિર ગણના ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    પાછા ggml_zdnn_graph_compute(backend, cgraph);
}

સ્થિર ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ખાલી,
    /* .get_tensor_async       = */ ખાલી,
    /* .set_tensor_2d_async    = */ ખાલી,
    /* .get_tensor_2d_async    = */ ખાલી,
    /* .cpy_tensor_async       = */ ખાલી,
    /* .synchronize            = */ ખાલી,
    /* .graph_plan_create      = */ ખાલી,
    /* .graph_plan_free        = */ ખાલી,
    /* .graph_plan_update      = */ ખાલી,
    /* .graph_plan_compute     = */ ખાલી,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ ખાલી,
    /* .event_wait             = */ ખાલી,
    /* .graph_optimize         = */ ખાલી,
};

સ્થિર ggml_guid_t ggml_backend_zdnn_guid(શૂન્ય) {
    સ્થિર અક્ષર guid_str[] = "IBM-ZDNN-ACCELER";
    પાછા reinterpret_cast<ggml_guid_t>(guid_str);
}

તાર્કિક ggml_backend_is_zdnn(ggml_backend_t backend) {
    પાછા backend != ખાલી &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

સ્થિર અચલ અક્ષર * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    પાછા GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

સ્થિર અચલ અક્ષર * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    પાછા "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    *મુક્ત  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    પાછા GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ ખોટું,
        /* .host_buffer          = */ ખોટું,
        /* .buffer_from_host_ptr = */ ખોટું,
        /* .events               = */ ખોટું,
        /* .mmap_support         = */ સાચું,
    };
}

સ્થિર ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    જો (ctx == ખાલી) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        પાછા ખાલી;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(કદ(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    પાછા backend;

    GGML_UNUSED(params);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    પાછા ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

સ્થિર તાર્કિક ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    પાછા ggml_zdnn_supports_op(ctx_dev, op);
}

સ્થિર તાર્કિક ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    પાછા
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

સ્થિર ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ ખાલી,
    /* .buffer_from_host_ptr = */ ખાલી,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ખાલી,
    /* .event_new            = */ ખાલી,
    /* .event_free           = */ ખાલી,
    /* .event_synchronize    = */ ખાલી,
};

//
// backend registry
//

સ્થિર અચલ અક્ષર * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    પાછા GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

સ્થિર size_t ggml_backend_zdnn_reg_device_ગણતરી(ggml_backend_reg_t reg) {
    જો (!zdnn_is_nnpa_installed()) {
        પાછા 0;
    }
    પાછા 1;

    GGML_UNUSED(reg);
}

સ્થિર ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    પાછા &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

સ્થિર ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ખાલી, ખાલી },
};

સ્થિર ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    પાછા g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

સ્થિર શૂન્ય ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, પૂર્ણાંક n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

સ્થિર શૂન્ય * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    જો (strcmp(name, "ggml_backend_get_features") == 0) {
        પાછા (શૂન્ય *) ggml_backend_zdnn_get_features;
    }
    જો (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        પાછા (શૂન્ય *) ggml_backend_zdnn_set_n_threads;
    }

    પાછા ખાલી;

    GGML_UNUSED(reg);
}

સ્થિર ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_ગણતરી,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

સ્થિર શૂન્ય ggml_zdnn_cleanup(શૂન્ય) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(શૂન્ય) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ ખાલી
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        પાછા &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
