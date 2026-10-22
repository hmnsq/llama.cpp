#ضمن "ggml-zdnn.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"

#ضمن "ggml-zdnn/common.hpp"
#ضمن "ggml-zdnn/fattn.hpp"
#ضمن "ggml-zdnn/mmf.hpp"
#ضمن "ggml-zdnn/mmq.hpp"
#ضمن "ggml-zdnn/utils.hpp"
#ضمن "ggml.h"

#ضمن <vector>
#ضمن <memory>
#ضمن <csignal>  // raise(SIGTRAP)
#ضمن <unistd.h>

ساكن فراغ ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];  // weights
    ثابت ggml_tensor * src1 = dst->src[1];  // inputs

    𐤀𐤌 (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } وإلا {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

ساكن منطقي ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    بدل (dst->op) {
        حالة GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } اكسر;

        حالة GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } اكسر;

        افتراضي:
            أرجع خطأ;
    }

    أرجع true;
}

ساكن تعداد ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    لكل (صحيح i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * node = gf->nodes[i];

        𐤀𐤌 (ggml_is_empty(node)
            || node->op == GGML_OP_NONE
            || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE
            || node->op == GGML_OP_TRANSPOSE) {
            استمر;
        }

        𐤀𐤌 ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            استمر;
        }

        منطقي ok = ggml_zdnn_compute_forward(ctx, node);
        𐤀𐤌 (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, node->name, ggml_op_name(node->op));
        }

        GGML_ASSERT(ok);
    }

    أرجع GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

ساكن منطقي ggml_zdnn_supports_op(ثابت ggml_backend_zdnn_device_context * ctx_dev, ثابت ggml_tensor * op) {
    بدل (op->op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_PERMUTE:
            أرجع true;

        حالة GGML_OP_MUL_MAT:
            {
                ثابت ggml_tensor * weights = op->src[0];
                ثابت ggml_tensor * inputs  = op->src[1];

                ثابت int64_t ne10 = inputs->ne[0];
                ثابت int64_t ne0  = op->ne[0];
                ثابت int64_t ne1  = op->ne[1];

                ثابت int64_t max_batch = ctx_dev->max_size;

                𐤀𐤌 (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        أرجع خطأ;
                }

                بدل (weights->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        أرجع inputs->view_src == nullptr;
                    حالة GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        أرجع ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    افتراضي:
                        أرجع خطأ;
                }
            } اكسر;

        حالة GGML_OP_FLASH_ATTN_EXT:
            أرجع ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        افتراضي:
            أرجع خطأ;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
ساكن ggml_backend_reg    g_ggml_backend_zdnn_reg;
ساكن ggml_backend_device g_ggml_backend_zdnn_device;

ساكن ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ خطأ,
    /* .has_parmblkformat_1   = */ خطأ,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

ساكن صحيح ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    أكد(ctx != عدم);

    𐤀𐤌 (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    𐤀𐤌 (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, حجم(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_count++;
    أرجع ctx->zdnn_device;
}

ساكن فراغ ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    أكد(ctx != عدم);
    أكد(ctx->zdnn_device_ref_count > 0);

    ctx->zdnn_device_ref_count--;
    𐤀𐤌 (ctx->zdnn_device_ref_count == 0) {
        𐤀𐤌 (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

ساكن ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = جديد ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    صحيح device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    أرجع ctx;
}

ساكن فراغ ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    لكل (ثابت auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

ساكن فراغ ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    لكل (ثابت auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        𐤀𐤌 (buf->extra != nullptr) حرر(buf->extra->data);
        𐤀𐤌 (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    𐤀𐤌 (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

ساكن فراغ * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    أرجع ctx->all_data;
}

ساكن تعداد ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    𐤀𐤌 (tensor->view_src != عدم) {
        أكد(tensor->view_src->buffer->buft == buffer->buft);
        أرجع GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    𐤀𐤌 (ggml_is_empty(tensor)) {
        أرجع GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    ثابت int64_t tsize = ggml_nbytes(tensor);
    صحيح buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    بدل (tensor->op) {
        حالة GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (فراغ *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (صحيح)حجم(" (bias)"), tensor->name);

                ثابت int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } اكسر;
        افتراضي:
            اكسر;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    أرجع GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

ساكن فراغ ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((حرف *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ساكن فراغ ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size) {
    memcpy((حرف *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    𐤀𐤌 (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        𐤀𐤌 (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        أرجع;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    𐤀𐤌 (extra == nullptr) {
        أرجع;
    }

    𐤀𐤌 (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        أرجع;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    𐤀𐤌 (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        أرجع;
    }
    𐤀𐤌 (extra->ztensor.is_transformed == خطأ) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

ساكن فراغ ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * tensor, فراغ * data, size_t offset, size_t size) {
    memcpy(data, (ثابت حرف *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

ساكن فراغ ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

ساكن فراغ ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    لكل (ثابت auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        𐤀𐤌 (buf->extra != nullptr) {
            حرر(buf->extra->data);
        }

        𐤀𐤌 (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    𐤀𐤌 (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

ساكن ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ عدم,
    /* .get_tensor_2d = */ عدم,
    /* .cpy_tensor    = */ عدم,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

ساكن ثابت حرف * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    أرجع GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

ساكن ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = جديد ggml_backend_zdnn_buffer_context();

    ثابت size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    𐤀𐤌 ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    صحيح device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = true;
    ctx->n_buffers = 1;

    𐤀𐤌 (ctx->all_data != عدم) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    𐤀𐤌 (size_aligned > 0 && (ctx->all_data == عدم)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        أرجع عدم;
    }

    أرجع ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

ساكن size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    أرجع 256;

    GGML_UNUSED(buft);
}

ساكن منطقي ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    أرجع خطأ;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(فراغ) {
    ساكن ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ عدم,
            /* .get_alloc_size      = */ عدم,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

ساكن ثابت حرف * ggml_backend_zdnn_name(ggml_backend_t backend) {
    أرجع GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

ساكن فراغ ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    حرر(backend);
}

ساكن تعداد ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    أرجع ggml_zdnn_graph_compute(backend, cgraph);
}

ساكن ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ عدم,
    /* .get_tensor_async       = */ عدم,
    /* .set_tensor_2d_async    = */ عدم,
    /* .get_tensor_2d_async    = */ عدم,
    /* .cpy_tensor_async       = */ عدم,
    /* .synchronize            = */ عدم,
    /* .graph_plan_create      = */ عدم,
    /* .graph_plan_free        = */ عدم,
    /* .graph_plan_update      = */ عدم,
    /* .graph_plan_compute     = */ عدم,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ عدم,
    /* .event_wait             = */ عدم,
    /* .graph_optimize         = */ عدم,
};

ساكن ggml_guid_t ggml_backend_zdnn_guid(فراغ) {
    ساكن حرف guid_str[] = "IBM-ZDNN-ACCELER";
    أرجع reinterpret_cast<ggml_guid_t>(guid_str);
}

منطقي ggml_backend_is_zdnn(ggml_backend_t backend) {
    أرجع backend != عدم &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

ساكن ثابت حرف * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    أرجع GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

ساكن ثابت حرف * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    أرجع "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

ساكن فراغ ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * حرر, size_t * total) {
    *حرر  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    أرجع GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

ساكن فراغ ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ خطأ,
        /* .host_buffer          = */ خطأ,
        /* .buffer_from_host_ptr = */ خطأ,
        /* .events               = */ خطأ,
        /* .mmap_support         = */ true,
    };
}

ساكن ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    𐤀𐤌 (ctx == عدم) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        أرجع عدم;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(حجم(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    أرجع backend;

    GGML_UNUSED(params);
}

ساكن ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    أرجع ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

ساكن منطقي ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    أرجع ggml_zdnn_supports_op(ctx_dev, op);
}

ساكن منطقي ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    أرجع
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

ساكن ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ عدم,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ عدم,
    /* .event_new            = */ عدم,
    /* .event_free           = */ عدم,
    /* .event_synchronize    = */ عدم,
};

//
// backend registry
//

ساكن ثابت حرف * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    أرجع GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

ساكن size_t ggml_backend_zdnn_reg_device_count(ggml_backend_reg_t reg) {
    𐤀𐤌 (!zdnn_is_nnpa_installed()) {
        أرجع 0;
    }
    أرجع 1;

    GGML_UNUSED(reg);
}

ساكن ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    أرجع &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

ساكن ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { عدم, عدم },
};

ساكن ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    أرجع g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

ساكن فراغ ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, صحيح n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

ساكن فراغ * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, ثابت حرف * name) {
    𐤀𐤌 (strcmp(name, "ggml_backend_get_features") == 0) {
        أرجع (فراغ *) ggml_backend_zdnn_get_features;
    }
    𐤀𐤌 (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        أرجع (فراغ *) ggml_backend_zdnn_set_n_threads;
    }

    أرجع عدم;

    GGML_UNUSED(reg);
}

ساكن ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_count,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

ساكن فراغ ggml_zdnn_cleanup(فراغ) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(فراغ) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ عدم
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        أرجع &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
