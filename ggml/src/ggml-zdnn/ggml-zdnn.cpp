#অন্তর্ভাব "ggml-zdnn.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব "ggml-zdnn/common.hpp"
#অন্তর্ভাব "ggml-zdnn/fattn.hpp"
#অন্তর্ভাব "ggml-zdnn/mmf.hpp"
#অন্তর্ভাব "ggml-zdnn/mmq.hpp"
#অন্তর্ভাব "ggml-zdnn/utils.hpp"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <vector>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <csignal>  // raise(SIGTRAP)
#অন্তর্ভাব <unistd.h>

স্থির শূন্য ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];  // weights
    ধ্রুব ggml_tensor * src1 = dst->src[1];  // inputs

    যদি (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } অন্যথা {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

স্থির তার্কিক ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    বিকল্প (dst->op) {
        পক্ষ GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } ভঙ্গ;

        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } ভঙ্গ;

        স্বভাব:
            প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির গণনা ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    প্রতি (পূর্ণাঙ্ক i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * গ্রন্থি = gf->nodes[i];

        যদি (ggml_is_empty(গ্রন্থি)
            || গ্রন্থি->op == GGML_OP_NONE
            || গ্রন্থি->op == GGML_OP_RESHAPE
            || গ্রন্থি->op == GGML_OP_VIEW
            || গ্রন্থি->op == GGML_OP_PERMUTE
            || গ্রন্থি->op == GGML_OP_TRANSPOSE) {
            অনুবর্ত;
        }

        যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            অনুবর্ত;
        }

        তার্কিক ok = ggml_zdnn_compute_forward(ctx, গ্রন্থি);
        যদি (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, গ্রন্থি->name, ggml_op_name(গ্রন্থি->op));
        }

        GGML_ASSERT(ok);
    }

    প্রত্যাগম GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

স্থির তার্কিক ggml_zdnn_supports_op(ধ্রুব ggml_backend_zdnn_device_context * ctx_dev, ধ্রুব ggml_tensor * op) {
    বিকল্প (op->op) {
        পক্ষ GGML_OP_NONE:
        পক্ষ GGML_OP_RESHAPE:
        পক্ষ GGML_OP_VIEW:
        পক্ষ GGML_OP_TRANSPOSE:
        পক্ষ GGML_OP_PERMUTE:
            প্রত্যাগম সত্য;

        পক্ষ GGML_OP_MUL_MAT:
            {
                ধ্রুব ggml_tensor * weights = op->src[0];
                ধ্রুব ggml_tensor * inputs  = op->src[1];

                ধ্রুব int64_t ne10 = inputs->ne[0];
                ধ্রুব int64_t ne0  = op->ne[0];
                ধ্রুব int64_t ne1  = op->ne[1];

                ধ্রুব int64_t max_batch = ctx_dev->max_size;

                যদি (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        প্রত্যাগম অসত্য;
                }

                বিকল্প (weights->type) {
                    পক্ষ GGML_TYPE_F32:
                    পক্ষ GGML_TYPE_F16:
                    পক্ষ GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        প্রত্যাগম inputs->view_src == nullptr;
                    পক্ষ GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        প্রত্যাগম ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    স্বভাব:
                        প্রত্যাগম অসত্য;
                }
            } ভঙ্গ;

        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            প্রত্যাগম ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        স্বভাব:
            প্রত্যাগম অসত্য;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
স্থির ggml_backend_reg    g_ggml_backend_zdnn_reg;
স্থির ggml_backend_device g_ggml_backend_zdnn_device;

স্থির ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ অসত্য,
    /* .has_parmblkformat_1   = */ অসত্য,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

স্থির পূর্ণাঙ্ক ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    প্রতিজ্ঞা(ctx != রিক্ত);

    যদি (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    যদি (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, পরিমাণ(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_গুনতি++;
    প্রত্যাগম ctx->zdnn_device;
}

স্থির শূন্য ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    প্রতিজ্ঞা(ctx != রিক্ত);
    প্রতিজ্ঞা(ctx->zdnn_device_ref_গুনতি > 0);

    ctx->zdnn_device_ref_গুনতি--;
    যদি (ctx->zdnn_device_ref_গুনতি == 0) {
        যদি (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

স্থির ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = নূতন ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    পূর্ণাঙ্ক device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    প্রত্যাগম ctx;
}

স্থির শূন্য ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    প্রতি (ধ্রুব auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

স্থির শূন্য ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    প্রতি (ধ্রুব auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        যদি (buf->extra != nullptr) মুক্ত(buf->extra->data);
        যদি (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    যদি (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

স্থির শূন্য * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    প্রত্যাগম ctx->all_data;
}

স্থির গণনা ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    যদি (tensor->view_src != রিক্ত) {
        প্রতিজ্ঞা(tensor->view_src->buffer->buft == buffer->buft);
        প্রত্যাগম GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    যদি (ggml_is_empty(tensor)) {
        প্রত্যাগম GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ধ্রুব int64_t tsize = ggml_nbytes(tensor);
    পূর্ণাঙ্ক buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    বিকল্প (tensor->op) {
        পক্ষ GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (শূন্য *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (পূর্ণাঙ্ক)পরিমাণ(" (bias)"), tensor->name);

                ধ্রুব int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } ভঙ্গ;
        স্বভাব:
            ভঙ্গ;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    প্রত্যাগম GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

স্থির শূন্য ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((বর্ণ *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ধ্রুব শূন্য * data, size_t offset, size_t size) {
    memcpy((বর্ণ *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    যদি (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        যদি (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        প্রত্যাগম;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    যদি (extra == nullptr) {
        প্রত্যাগম;
    }

    যদি (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        প্রত্যাগম;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    যদি (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        প্রত্যাগম;
    }
    যদি (extra->ztensor.is_transformed == অসত্য) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, ধ্রুব ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    memcpy(data, (ধ্রুব বর্ণ *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

স্থির শূন্য ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    প্রতি (ধ্রুব auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        যদি (buf->extra != nullptr) {
            মুক্ত(buf->extra->data);
        }

        যদি (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    যদি (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

স্থির ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ রিক্ত,
    /* .get_tensor_2d = */ রিক্ত,
    /* .cpy_tensor    = */ রিক্ত,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

স্থির ধ্রুব বর্ণ * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = নূতন ggml_backend_zdnn_buffer_context();

    ধ্রুব size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    যদি ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    পূর্ণাঙ্ক device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = সত্য;
    ctx->n_buffers = 1;

    যদি (ctx->all_data != রিক্ত) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    যদি (size_aligned > 0 && (ctx->all_data == রিক্ত)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        প্রত্যাগম রিক্ত;
    }

    প্রত্যাগম ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

স্থির size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম 256;

    GGML_UNUSED(buft);
}

স্থির তার্কিক ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    প্রত্যাগম অসত্য;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(শূন্য) {
    স্থির ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ রিক্ত,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ রিক্ত,
            /* .get_alloc_size      = */ রিক্ত,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ রিক্ত,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ রিক্ত,
    };

    প্রত্যাগম &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

স্থির ধ্রুব বর্ণ * ggml_backend_zdnn_name(ggml_backend_t backend) {
    প্রত্যাগম GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    মুক্ত(backend);
}

স্থির গণনা ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    প্রত্যাগম ggml_zdnn_graph_compute(backend, cgraph);
}

স্থির ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ রিক্ত,
    /* .get_tensor_async       = */ রিক্ত,
    /* .set_tensor_2d_async    = */ রিক্ত,
    /* .get_tensor_2d_async    = */ রিক্ত,
    /* .cpy_tensor_async       = */ রিক্ত,
    /* .synchronize            = */ রিক্ত,
    /* .graph_plan_create      = */ রিক্ত,
    /* .graph_plan_free        = */ রিক্ত,
    /* .graph_plan_update      = */ রিক্ত,
    /* .graph_plan_compute     = */ রিক্ত,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ রিক্ত,
    /* .event_wait             = */ রিক্ত,
    /* .graph_optimize         = */ রিক্ত,
};

স্থির ggml_guid_t ggml_backend_zdnn_guid(শূন্য) {
    স্থির বর্ণ guid_str[] = "IBM-ZDNN-ACCELER";
    প্রত্যাগম reinterpret_cast<ggml_guid_t>(guid_str);
}

তার্কিক ggml_backend_is_zdnn(ggml_backend_t backend) {
    প্রত্যাগম backend != রিক্ত &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

স্থির ধ্রুব বর্ণ * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    প্রত্যাগম GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

স্থির ধ্রুব বর্ণ * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    প্রত্যাগম "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    *মুক্ত  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    প্রত্যাগম GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ অসত্য,
        /* .host_buffer          = */ অসত্য,
        /* .buffer_from_host_ptr = */ অসত্য,
        /* .events               = */ অসত্য,
        /* .mmap_support         = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ধ্রুব বর্ণ * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    যদি (ctx == রিক্ত) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        প্রত্যাগম রিক্ত;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(পরিমাণ(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    প্রত্যাগম backend;

    GGML_UNUSED(params);
}

স্থির ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    প্রত্যাগম ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

স্থির তার্কিক ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ধ্রুব ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    প্রত্যাগম ggml_zdnn_supports_op(ctx_dev, op);
}

স্থির তার্কিক ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    প্রত্যাগম
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

স্থির ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ রিক্ত,
    /* .buffer_from_host_ptr = */ রিক্ত,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ রিক্ত,
    /* .event_new            = */ রিক্ত,
    /* .event_free           = */ রিক্ত,
    /* .event_synchronize    = */ রিক্ত,
};

//
// backend registry
//

স্থির ধ্রুব বর্ণ * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    প্রত্যাগম GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_zdnn_reg_device_গুনতি(ggml_backend_reg_t reg) {
    যদি (!zdnn_is_nnpa_installed()) {
        প্রত্যাগম 0;
    }
    প্রত্যাগম 1;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    প্রত্যাগম &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

স্থির ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { রিক্ত, রিক্ত },
};

স্থির ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    প্রত্যাগম g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

স্থির শূন্য ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, পূর্ণাঙ্ক n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

স্থির শূন্য * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, ধ্রুব বর্ণ * name) {
    যদি (strcmp(name, "ggml_backend_get_features") == 0) {
        প্রত্যাগম (শূন্য *) ggml_backend_zdnn_get_features;
    }
    যদি (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        প্রত্যাগম (শূন্য *) ggml_backend_zdnn_set_n_threads;
    }

    প্রত্যাগম রিক্ত;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_গুনতি,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

স্থির শূন্য ggml_zdnn_cleanup(শূন্য) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(শূন্য) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ রিক্ত
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        প্রত্যাগম &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
