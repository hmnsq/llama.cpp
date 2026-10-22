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

    ধ্রুবক ggml_tensor * src0 = dst->src[0];  // weights
    ধ্রুবক ggml_tensor * src1 = dst->src[1];  // inputs

    যদি (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } অন্যথা {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

স্থির যৌক্তিক ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    নির্বাচন (dst->op) {
        ক্ষেত্র GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } বিরতি;

        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } বিরতি;

        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }

    ফেরত সত্য;
}

স্থির গণনা ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    জন্য (পূর্ণসংখ্যা i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * গ্রন্থি = gf->nodes[i];

        যদি (ggml_is_empty(গ্রন্থি)
            || গ্রন্থি->op == GGML_OP_NONE
            || গ্রন্থি->op == GGML_OP_RESHAPE
            || গ্রন্থি->op == GGML_OP_VIEW
            || গ্রন্থি->op == GGML_OP_PERMUTE
            || গ্রন্থি->op == GGML_OP_TRANSPOSE) {
            চালু;
        }

        যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            চালু;
        }

        যৌক্তিক ok = ggml_zdnn_compute_forward(ctx, গ্রন্থি);
        যদি (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, গ্রন্থি->name, ggml_op_name(গ্রন্থি->op));
        }

        GGML_ASSERT(ok);
    }

    ফেরত GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

স্থির যৌক্তিক ggml_zdnn_supports_op(ধ্রুবক ggml_backend_zdnn_device_context * ctx_dev, ধ্রুবক ggml_tensor * op) {
    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_NONE:
        ক্ষেত্র GGML_OP_RESHAPE:
        ক্ষেত্র GGML_OP_VIEW:
        ক্ষেত্র GGML_OP_TRANSPOSE:
        ক্ষেত্র GGML_OP_PERMUTE:
            ফেরত সত্য;

        ক্ষেত্র GGML_OP_MUL_MAT:
            {
                ধ্রুবক ggml_tensor * weights = op->src[0];
                ধ্রুবক ggml_tensor * inputs  = op->src[1];

                ধ্রুবক int64_t ne10 = inputs->ne[0];
                ধ্রুবক int64_t ne0  = op->ne[0];
                ধ্রুবক int64_t ne1  = op->ne[1];

                ধ্রুবক int64_t max_batch = ctx_dev->max_size;

                যদি (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        ফেরত মিথ্যা;
                }

                নির্বাচন (weights->type) {
                    ক্ষেত্র GGML_TYPE_F32:
                    ক্ষেত্র GGML_TYPE_F16:
                    ক্ষেত্র GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        ফেরত inputs->view_src == nullptr;
                    ক্ষেত্র GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        ফেরত ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    পূর্বনির্ধারিত:
                        ফেরত মিথ্যা;
                }
            } বিরতি;

        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            ফেরত ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
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
    /* .has_parmblkformat_0   = */ মিথ্যা,
    /* .has_parmblkformat_1   = */ মিথ্যা,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

স্থির পূর্ণসংখ্যা ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    প্রতিজ্ঞা(ctx != ফাঁকা);

    যদি (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    যদি (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_আকার();
        strncpy(ctx->name, GGML_ZDNN_NAME, আকার(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_গুনতি++;
    ফেরত ctx->zdnn_device;
}

স্থির শূন্য ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    প্রতিজ্ঞা(ctx != ফাঁকা);
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

    ggml_backend_zdnn_context * ctx = নতুন ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    পূর্ণসংখ্যা device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    ফেরত ctx;
}

স্থির শূন্য ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    জন্য (ধ্রুবক auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_বাফার(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

স্থির শূন্য ggml_backend_zdnn_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    ggml_backend_zdnn_বাফার_context * ctx = (ggml_backend_zdnn_বাফার_context *)বাফার->context;

    জন্য (ধ্রুবক auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_বাফার * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        যদি (buf->extra != nullptr) মুক্ত(buf->extra->তথ্য);
        যদি (buf->ztensor.বাফার_আকার > 0) ZDNN_CHECK(zdnn_free_ztensor_বাফার(&buf->ztensor));
    }

    যদি (ctx->owned) {
        ggml_aligned_free(ctx->all_তথ্য, ctx->all_আকার);
    }

    delete ctx;
}

স্থির শূন্য * ggml_backend_zdnn_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    ggml_backend_zdnn_বাফার_context * ctx = (ggml_backend_zdnn_বাফার_context *)বাফার->context;
    ফেরত ctx->all_তথ্য;
}

স্থির গণনা ggml_status ggml_backend_zdnn_বাফার_init_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor) {
    যদি (tensor->view_src != ফাঁকা) {
        প্রতিজ্ঞা(tensor->view_src->বাফার->buft == বাফার->buft);
        ফেরত GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    যদি (ggml_is_empty(tensor)) {
        ফেরত GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_বাফার_context * ctx = (ggml_backend_zdnn_বাফার_context *)বাফার->context;

    ধ্রুবক int64_t tsize = ggml_nbytes(tensor);
    পূর্ণসংখ্যা বাফার_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_বাফার> zdnn_বাফার = std::make_unique<ggml_backend_zdnn_বাফার>();
    zdnn_বাফার->তথ্য = tensor->তথ্য;
    zdnn_বাফার->size = tsize;
    zdnn_বাফার->extra = nullptr;
    snprintf(zdnn_বাফার->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_বাফার.get(), tensor);
    tensor->extra = zdnn_বাফার.get();

    নির্বাচন (tensor->op) {
        ক্ষেত্র GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_বাফার> zdnn_bias_বাফার = std::make_unique<ggml_backend_zdnn_বাফার>();
                zdnn_bias_বাফার->তথ্য = (শূন্য *)calloc(tensor->ne[0], ggml_element_আকার(tensor));
                zdnn_bias_বাফার->size = ggml_element_আকার(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_বাফার->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (পূর্ণসংখ্যা)আকার(" (bias)"), tensor->name);

                ধ্রুবক int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_বাফার->pre_tfm_desc,
                                        zdnn_bias_বাফার->tfm_desc,
                                        zdnn_bias_বাফার->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_বাফার->ztensor, zdnn_bias_বাফার->তথ্য);
                zdnn_বাফার->extra = zdnn_bias_বাফার.get();

                ctx->buffers.push_back(std::move(zdnn_bias_বাফার));
                ctx->n_buffers++;
            } বিরতি;
        পূর্বনির্ধারিত:
            বিরতি;
    }

    ctx->buffers.push_back(std::move(zdnn_বাফার));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    ফেরত GGML_STATUS_SUCCESS;

    GGML_UNUSED(বাফার_idx);
}

স্থির শূন্য ggml_backend_zdnn_বাফার_memset_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((অক্ষর *)tensor->তথ্য + offset, value, size);

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_zdnn_বাফার_set_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    memcpy((অক্ষর *)tensor->তথ্য + offset, তথ্য, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    যদি (tensor->view_src != nullptr) {
        ggml_backend_zdnn_বাফার * src_extra = (ggml_backend_zdnn_বাফার *)tensor->view_src->extra;
        যদি (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        ফেরত;
    }

    ggml_backend_zdnn_বাফার * extra = (ggml_backend_zdnn_বাফার *)tensor->extra;
    যদি (extra == nullptr) {
        ফেরত;
    }

    যদি (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        ফেরত;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    যদি (tensor->বাফার->usage != GGML_BACKEND_বাফার_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        ফেরত;
    }
    যদি (extra->ztensor.is_transformed == মিথ্যা) ggml_zdnn_load_tensor(extra->ztensor, tensor->তথ্য);

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_zdnn_বাফার_get_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    memcpy(তথ্য, (ধ্রুবক অক্ষর *)tensor->তথ্য + offset, size);

    GGML_UNUSED(বাফার);
}

স্থির শূন্য ggml_backend_zdnn_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    ggml_backend_zdnn_বাফার_context * ctx = (ggml_backend_zdnn_বাফার_context *)বাফার->context;

    memset(ctx->all_তথ্য, value, ctx->all_আকার);
}

স্থির শূন্য ggml_backend_zdnn_বাফার_reset(ggml_backend_বাফার_t বাফার) {
    ggml_backend_zdnn_বাফার_context * ctx = (ggml_backend_zdnn_বাফার_context *)বাফার->context;

    জন্য (ধ্রুবক auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_বাফার * buf = buf_ptr.get();

        যদি (buf->extra != nullptr) {
            মুক্ত(buf->extra->তথ্য);
        }

        যদি (buf->ztensor.বাফার_আকার > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_বাফার(&buf->ztensor));
        }
    }

    যদি (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

স্থির ggml_backend_বাফার_i ggml_backend_zdnn_বাফার_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_বাফার_free_বাফার,
    /* .get_base      = */ ggml_backend_zdnn_বাফার_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_বাফার_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_বাফার_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_বাফার_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_বাফার_get_tensor,
    /* .set_tensor_2d = */ ফাঁকা,
    /* .get_tensor_2d = */ ফাঁকা,
    /* .cpy_tensor    = */ ফাঁকা,
    /* .clear         = */ ggml_backend_zdnn_বাফার_clear,
    /* .reset         = */ ggml_backend_zdnn_বাফার_reset,
};

//
// default buffer type
//

স্থির ধ্রুবক অক্ষর * ggml_backend_zdnn_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    ফেরত GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

স্থির ggml_backend_বাফার_t ggml_backend_zdnn_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    ggml_backend_zdnn_বাফার_context * ctx = নতুন ggml_backend_zdnn_বাফার_context();

    ধ্রুবক size_t আকার_page = sysconf(_SC_PAGESIZE);

    size_t আকার_aligned = size;
    যদি ((আকার_aligned % আকার_page) != 0) {
        আকার_aligned += আকার_page - (আকার_aligned % আকার_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    পূর্ণসংখ্যা device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_তথ্য  = ggml_aligned_malloc(আকার_aligned);
    ctx->all_আকার  = আকার_aligned;
    ctx->owned     = সত্য;
    ctx->n_buffers = 1;

    যদি (ctx->all_তথ্য != ফাঁকা) {
        std::unique_ptr<ggml_backend_zdnn_বাফার> zdnn_বাফার = std::make_unique<ggml_backend_zdnn_বাফার>();
        zdnn_বাফার->তথ্য = ctx->all_তথ্য;
        zdnn_বাফার->size = আকার_aligned;
        ctx->buffers.push_back(std::move(zdnn_বাফার));
    }

    যদি (আকার_aligned > 0 && (ctx->all_তথ্য == ফাঁকা)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, আকার_aligned / 1024.0 / 1024.0);
        delete ctx;
        ফেরত ফাঁকা;
    }

    ফেরত ggml_backend_বাফার_init(buft, ggml_backend_zdnn_বাফার_i, ctx, size);
}

স্থির size_t ggml_backend_zdnn_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    ফেরত 256;

    GGML_UNUSED(buft);
}

স্থির যৌক্তিক ggml_backend_zdnn_বাফার_type_is_host(ggml_backend_বাফার_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    ফেরত মিথ্যা;

    GGML_UNUSED(buft);
}

ggml_backend_বাফার_type_t ggml_backend_zdnn_বাফার_type(শূন্য) {
    স্থির ggml_backend_বাফার_type ggml_backend_বাফার_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_বাফার_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_বাফার_type_alloc_বাফার,
            /* .alloc_buffer_n      = */ ফাঁকা,
            /* .get_alignment       = */ ggml_backend_zdnn_বাফার_type_get_alignment,
            /* .get_max_size        = */ ফাঁকা,
            /* .get_alloc_size      = */ ফাঁকা,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ফাঁকা,
            /* .is_host             = */ ggml_backend_zdnn_বাফার_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ ফাঁকা,
    };

    ফেরত &ggml_backend_বাফার_type_zdnn;
}

//
// backend
//

স্থির ধ্রুবক অক্ষর * ggml_backend_zdnn_name(ggml_backend_t backend) {
    ফেরত GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    মুক্ত(backend);
}

স্থির গণনা ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ফেরত ggml_zdnn_graph_compute(backend, cgraph);
}

স্থির ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ ফাঁকা,
    /* .get_tensor_async       = */ ফাঁকা,
    /* .set_tensor_2d_async    = */ ফাঁকা,
    /* .get_tensor_2d_async    = */ ফাঁকা,
    /* .cpy_tensor_async       = */ ফাঁকা,
    /* .synchronize            = */ ফাঁকা,
    /* .graph_plan_create      = */ ফাঁকা,
    /* .graph_plan_free        = */ ফাঁকা,
    /* .graph_plan_update      = */ ফাঁকা,
    /* .graph_plan_compute     = */ ফাঁকা,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ ফাঁকা,
    /* .event_wait             = */ ফাঁকা,
    /* .graph_optimize         = */ ফাঁকা,
};

স্থির ggml_guid_t ggml_backend_zdnn_guid(শূন্য) {
    স্থির অক্ষর guid_str[] = "IBM-ZDNN-ACCELER";
    ফেরত reinterpret_cast<ggml_guid_t>(guid_str);
}

যৌক্তিক ggml_backend_is_zdnn(ggml_backend_t backend) {
    ফেরত backend != ফাঁকা &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

স্থির ধ্রুবক অক্ষর * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    ফেরত GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

স্থির ধ্রুবক অক্ষর * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    ফেরত "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    *মুক্ত  = 0;
    *যোগ = 0;

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    ফেরত GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);
    props->caps = {
        /* .async                = */ মিথ্যা,
        /* .host_buffer          = */ মিথ্যা,
        /* .buffer_from_host_ptr = */ মিথ্যা,
        /* .events               = */ মিথ্যা,
        /* .mmap_support         = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    যদি (ctx == ফাঁকা) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ফেরত ফাঁকা;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(আকার(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    ফেরত backend;

    GGML_UNUSED(params);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_zdnn_device_get_বাফার_type(ggml_backend_dev_t dev) {
    ফেরত ggml_backend_zdnn_বাফার_type();

    GGML_UNUSED(dev);
}

স্থির যৌক্তিক ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    ফেরত ggml_zdnn_supports_op(ctx_dev, op);
}

স্থির যৌক্তিক ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    ফেরত
        buft->iface.get_name == ggml_backend_zdnn_বাফার_type_get_name;

    GGML_UNUSED(dev);
}

স্থির ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_বাফার_type,
    /* .get_host_buffer_type = */ ফাঁকা,
    /* .buffer_from_host_ptr = */ ফাঁকা,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ ফাঁকা,
    /* .event_new            = */ ফাঁকা,
    /* .event_free           = */ ফাঁকা,
    /* .event_synchronize    = */ ফাঁকা,
};

//
// backend registry
//

স্থির ধ্রুবক অক্ষর * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    ফেরত GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_zdnn_reg_device_গুনতি(ggml_backend_reg_t reg) {
    যদি (!zdnn_is_nnpa_installed()) {
        ফেরত 0;
    }
    ফেরত 1;

    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t সূচক) {
    GGML_ASSERT(সূচক == 0);

    ফেরত &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(সূচক);
}

স্থির ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { ফাঁকা, ফাঁকা },
};

স্থির ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    ফেরত g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

স্থির শূন্য ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, পূর্ণসংখ্যা n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

স্থির শূন্য * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, ধ্রুবক অক্ষর * name) {
    যদি (strcmp(name, "ggml_backend_get_features") == 0) {
        ফেরত (শূন্য *) ggml_backend_zdnn_get_features;
    }
    যদি (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ফেরত (শূন্য *) ggml_backend_zdnn_set_n_threads;
    }

    ফেরত ফাঁকা;

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
            /* .context     = */ ফাঁকা
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        ফেরত &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
