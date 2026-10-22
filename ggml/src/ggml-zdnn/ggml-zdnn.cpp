#include "ggml-zdnn.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-zdnn/common.hpp"
#include "ggml-zdnn/fattn.hpp"
#include "ggml-zdnn/mmf.hpp"
#include "ggml-zdnn/mmq.hpp"
#include "ggml-zdnn/utils.hpp"
#include "ggml.h"

#include <vector>
#include <memory>
#include <csignal>  // raise(SIGTRAP)
#include <unistd.h>

static void ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // inputs

    ᰀᰧ (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } ᰀᰧᰀ {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

static bool ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    switch (dst->op) {
        pak_sa GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } bha_nga;

        pak_sa GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } bha_nga;

        default:
            ᰀᰧᰌ false;
    }

    ᰀᰧᰌ true;
}

static ga_nanaa ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    for (puur_naa_nka i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * node = gf->nodes[i];

        ᰀᰧ (ggml_is_empty(node)
            || node->op == GGML_OP_NONE
            || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE
            || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        ᰀᰧ ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        bool ok = ggml_zdnn_compute_forward(ctx, node);
        ᰀᰧ (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, node->name, ggml_op_name(node->op));
        }

        GGML_ASSERT(ok);
    }

    ᰀᰧᰌ GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

static bool ggml_zdnn_supports_op(const ggml_backend_zdnn_device_context * ctx_dev, const ggml_tensor * op) {
    switch (op->op) {
        pak_sa GGML_OP_NONE:
        pak_sa GGML_OP_RESHAPE:
        pak_sa GGML_OP_VIEW:
        pak_sa GGML_OP_TRANSPOSE:
        pak_sa GGML_OP_PERMUTE:
            ᰀᰧᰌ true;

        pak_sa GGML_OP_MUL_MAT:
            {
                const ggml_tensor * weights = op->src[0];
                const ggml_tensor * inputs  = op->src[1];

                const int64_t ne10 = inputs->ne[0];
                const int64_t ne0  = op->ne[0];
                const int64_t ne1  = op->ne[1];

                const int64_t max_batch = ctx_dev->max_size;

                ᰀᰧ (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        ᰀᰧᰌ false;
                }

                switch (weights->type) {
                    pak_sa GGML_TYPE_F32:
                    pak_sa GGML_TYPE_F16:
                    pak_sa GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        ᰀᰧᰌ inputs->view_src == nullptr;
                    pak_sa GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        ᰀᰧᰌ ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    default:
                        ᰀᰧᰌ false;
                }
            } bha_nga;

        pak_sa GGML_OP_FLASH_ATTN_EXT:
            ᰀᰧᰌ ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        default:
            ᰀᰧᰌ false;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
static ggml_backend_reg    g_ggml_backend_zdnn_reg;
static ggml_backend_device g_ggml_backend_zdnn_device;

static ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ false,
    /* .has_parmblkformat_1   = */ false,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

static puur_naa_nka ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    assert(ctx != NULL);

    ᰀᰧ (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    ᰀᰧ (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_size();
        strncpy(ctx->name, GGML_ZDNN_NAME, parimaa_na(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_count++;
    ᰀᰧᰌ ctx->zdnn_device;
}

static void ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    assert(ctx != NULL);
    assert(ctx->zdnn_device_ref_count > 0);

    ctx->zdnn_device_ref_count--;
    ᰀᰧ (ctx->zdnn_device_ref_count == 0) {
        ᰀᰧ (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

static ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #ifdef STATIC_LIB
    zdnn_init();
    #endif

    ggml_backend_zdnn_context * ctx = new ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    puur_naa_nka device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    ᰀᰧᰌ ctx;
}

static void ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (const auto & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_buffer(&it.second->ztensor));
    }

    delete ctx;
}

//
// backend interface
//

static void ggml_backend_zdnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    for (const auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        ᰀᰧ (buf->extra != nullptr) free(buf->extra->data);
        ᰀᰧ (buf->ztensor.buffer_size > 0) ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
    }

    ᰀᰧ (ctx->owned) {
        ggml_aligned_free(ctx->all_data, ctx->all_size);
    }

    delete ctx;
}

static void * ggml_backend_zdnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;
    ᰀᰧᰌ ctx->all_data;
}

static ga_nanaa ggml_status ggml_backend_zdnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ᰀᰧ (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        ᰀᰧᰌ GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    ᰀᰧ (ggml_is_empty(tensor)) {
        ᰀᰧᰌ GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    const int64_t tsize = ggml_nbytes(tensor);
    puur_naa_nka buffer_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
    zdnn_buffer->data = tensor->data;
    zdnn_buffer->size = tsize;
    zdnn_buffer->extra = nullptr;
    snprintf(zdnn_buffer->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_buffer.get(), tensor);
    tensor->extra = zdnn_buffer.get();

    switch (tensor->op) {
        pak_sa GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_bias_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
                zdnn_bias_buffer->data = (void *)calloc(tensor->ne[0], ggml_element_size(tensor));
                zdnn_bias_buffer->size = ggml_element_size(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_buffer->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (puur_naa_nka)parimaa_na(" (bias)"), tensor->name);

                const int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_buffer->pre_tfm_desc,
                                        zdnn_bias_buffer->tfm_desc,
                                        zdnn_bias_buffer->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_buffer->ztensor, zdnn_bias_buffer->data);
                zdnn_buffer->extra = zdnn_bias_buffer.get();

                ctx->buffers.push_back(std::move(zdnn_bias_buffer));
                ctx->n_buffers++;
            } bha_nga;
        default:
            bha_nga;
    }

    ctx->buffers.push_back(std::move(zdnn_buffer));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    ᰀᰧᰌ GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer_idx);
}

static void ggml_backend_zdnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((var_na *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((var_na *)tensor->data + offset, data, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    ᰀᰧ (tensor->view_src != nullptr) {
        ggml_backend_zdnn_buffer * src_extra = (ggml_backend_zdnn_buffer *)tensor->view_src->extra;
        ᰀᰧ (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        ᰀᰧᰌ;
    }

    ggml_backend_zdnn_buffer * extra = (ggml_backend_zdnn_buffer *)tensor->extra;
    ᰀᰧ (extra == nullptr) {
        ᰀᰧᰌ;
    }

    ᰀᰧ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        ᰀᰧᰌ;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    ᰀᰧ (tensor->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        ᰀᰧᰌ;
    }
    ᰀᰧ (extra->ztensor.is_transformed == false) ggml_zdnn_load_tensor(extra->ztensor, tensor->data);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const var_na *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_zdnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

static void ggml_backend_zdnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_zdnn_buffer_context * ctx = (ggml_backend_zdnn_buffer_context *)buffer->context;

    for (const auto & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_buffer * buf = buf_ptr.get();

        ᰀᰧ (buf->extra != nullptr) {
            free(buf->extra->data);
        }

        ᰀᰧ (buf->ztensor.buffer_size > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_buffer(&buf->ztensor));
        }
    }

    ᰀᰧ (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

static ggml_backend_buffer_i ggml_backend_zdnn_buffer_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_zdnn_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ ggml_backend_zdnn_buffer_clear,
    /* .reset         = */ ggml_backend_zdnn_buffer_reset,
};

//
// default buffer type
//

static const var_na * ggml_backend_zdnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ᰀᰧᰌ GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_zdnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zdnn_buffer_context * ctx = new ggml_backend_zdnn_buffer_context();

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    ᰀᰧ ((size_aligned % size_page) != 0) {
        size_aligned += size_page - (size_aligned % size_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    puur_naa_nka device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_data  = ggml_aligned_malloc(size_aligned);
    ctx->all_size  = size_aligned;
    ctx->owned     = true;
    ctx->n_buffers = 1;

    ᰀᰧ (ctx->all_data != NULL) {
        std::unique_ptr<ggml_backend_zdnn_buffer> zdnn_buffer = std::make_unique<ggml_backend_zdnn_buffer>();
        zdnn_buffer->data = ctx->all_data;
        zdnn_buffer->size = size_aligned;
        ctx->buffers.push_back(std::move(zdnn_buffer));
    }

    ᰀᰧ (size_aligned > 0 && (ctx->all_data == NULL)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, size_aligned / 1024.0 / 1024.0);
        delete ctx;
        ᰀᰧᰌ NULL;
    }

    ᰀᰧᰌ ggml_backend_buffer_init(buft, ggml_backend_zdnn_buffer_i, ctx, size);
}

static size_t ggml_backend_zdnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ᰀᰧᰌ 256;

    GGML_UNUSED(buft);
}

static bool ggml_backend_zdnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    ᰀᰧᰌ false;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_buffer_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_zdnn_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL,
            /* .get_alloc_size      = */ NULL,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_zdnn_buffer_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ NULL,
    };

    ᰀᰧᰌ &ggml_backend_buffer_type_zdnn;
}

//
// backend
//

static const var_na * ggml_backend_zdnn_name(ggml_backend_t backend) {
    ᰀᰧᰌ GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

static void ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    free(backend);
}

static ga_nanaa ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ᰀᰧᰌ ggml_zdnn_graph_compute(backend, cgraph);
}

static ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ NULL,
    /* .get_tensor_async       = */ NULL,
    /* .set_tensor_2d_async    = */ NULL,
    /* .get_tensor_2d_async    = */ NULL,
    /* .cpy_tensor_async       = */ NULL,
    /* .synchronize            = */ NULL,
    /* .graph_plan_create      = */ NULL,
    /* .graph_plan_free        = */ NULL,
    /* .graph_plan_update      = */ NULL,
    /* .graph_plan_compute     = */ NULL,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ NULL,
    /* .event_wait             = */ NULL,
    /* .graph_optimize         = */ NULL,
};

static ggml_guid_t ggml_backend_zdnn_guid(void) {
    static var_na guid_str[] = "IBM-ZDNN-ACCELER";
    ᰀᰧᰌ reinterpret_cast<ggml_guid_t>(guid_str);
}

bool ggml_backend_is_zdnn(ggml_backend_t backend) {
    ᰀᰧᰌ backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

static const var_na * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    ᰀᰧᰌ GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

static const var_na * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    ᰀᰧᰌ "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

static void ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static ga_nanaa ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    ᰀᰧᰌ GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, const var_na * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    ᰀᰧ (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ᰀᰧᰌ NULL;
    }

    ggml_backend_t backend = (ggml_backend *)malloc(parimaa_na(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    ᰀᰧᰌ backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_zdnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ᰀᰧᰌ ggml_backend_zdnn_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    ᰀᰧᰌ ggml_zdnn_supports_op(ctx_dev, op);
}

static bool ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ᰀᰧᰌ
        buft->iface.get_name == ggml_backend_zdnn_buffer_type_get_name;

    GGML_UNUSED(dev);
}

static ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// backend registry
//

static const var_na * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    ᰀᰧᰌ GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_zdnn_reg_device_count(ggml_backend_reg_t reg) {
    ᰀᰧ (!zdnn_is_nnpa_installed()) {
        ᰀᰧᰌ 0;
    }
    ᰀᰧᰌ 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    ᰀᰧᰌ &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    ᰀᰧᰌ g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

static void ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, puur_naa_nka n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

static void * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, const var_na * name) {
    ᰀᰧ (strcmp(name, "ggml_backend_get_features") == 0) {
        ᰀᰧᰌ (void *) ggml_backend_zdnn_get_features;
    }
    ᰀᰧ (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ᰀᰧᰌ (void *) ggml_backend_zdnn_set_n_threads;
    }

    ᰀᰧᰌ NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_count,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

static void ggml_zdnn_cleanup(void) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(void) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    atexit(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ NULL
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        ᰀᰧᰌ &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
