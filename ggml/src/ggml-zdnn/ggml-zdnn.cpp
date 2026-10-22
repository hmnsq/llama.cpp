#समावेश "ggml-zdnn.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-backend-impl.h"

#समावेश "ggml-zdnn/common.hpp"
#समावेश "ggml-zdnn/fattn.hpp"
#समावेश "ggml-zdnn/mmf.hpp"
#समावेश "ggml-zdnn/mmq.hpp"
#समावेश "ggml-zdnn/utils.hpp"
#समावेश "ggml.h"

#समावेश <vector>
#समावेश <memory>
#समावेश <csignal>  // raise(SIGTRAP)
#समावेश <unistd.h>

अटल व्योम ggml_zdnn_compute_forward_mul_mat(
    ggml_backend_zdnn_context * ctx,
          ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];  // weights
    स्थिर ggml_tensor * src1 = dst->src[1];  // inputs

    अगर (ggml_is_quantized(src0->type)) {
        ggml_zdnn_mul_mat_q(ctx, src0, src1, dst);
    } अन्यथा {
        ggml_zdnn_mul_mat_f(ctx, src0, src1, dst);
    }
}

अटल bool ggml_zdnn_compute_forward(
    ggml_backend_zdnn_context * ctx,
    ggml_tensor * dst) {

    चयन (dst->op) {
        हाल GGML_OP_MUL_MAT:
            {
                ggml_zdnn_compute_forward_mul_mat(ctx, dst);
            } अवरोध;

        हाल GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_zdnn_flash_attn_ext(ctx, dst);
            } अवरोध;

        शेष:
            वापस false;
    }

    वापस true;
}

अटल क्रमागत ggml_status ggml_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_backend_zdnn_context        * ctx     = (       ggml_backend_zdnn_context *)backend->context;
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)backend->device->context;

    ctx->gf = gf;
    क्रम (पूर्णांक i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * गाँठ = gf->nodes[i];

        अगर (ggml_is_empty(गाँठ)
            || गाँठ->op == GGML_OP_NONE
            || गाँठ->op == GGML_OP_RESHAPE
            || गाँठ->op == GGML_OP_VIEW
            || गाँठ->op == GGML_OP_PERMUTE
            || गाँठ->op == GGML_OP_TRANSPOSE) {
            जारी;
        }

        अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;
        }

        bool ok = ggml_zdnn_compute_forward(ctx, गाँठ);
        अगर (!ok) {
            GGML_LOG_ERROR("%s: unsupported op %s (%s)\n",
                           __func__, गाँठ->name, ggml_op_name(गाँठ->op));
        }

        GGML_ASSERT(ok);
    }

    वापस GGML_STATUS_SUCCESS;

    GGML_UNUSED(ctx_dev);
}

अटल bool ggml_zdnn_supports_op(स्थिर ggml_backend_zdnn_device_context * ctx_dev, स्थिर ggml_tensor * op) {
    चयन (op->op) {
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_PERMUTE:
            वापस true;

        हाल GGML_OP_MUL_MAT:
            {
                स्थिर ggml_tensor * weights = op->src[0];
                स्थिर ggml_tensor * inputs  = op->src[1];

                स्थिर int64_t ne10 = inputs->ne[0];
                स्थिर int64_t ne0  = op->ne[0];
                स्थिर int64_t ne1  = op->ne[1];

                स्थिर int64_t max_batch = ctx_dev->max_size;

                अगर (!ggml_is_matrix(weights) || !ggml_is_matrix(inputs) ||
                    !ggml_is_contiguous(weights) || !ggml_is_contiguous(inputs) ||
                    weights->view_src != nullptr ||
                    ne0 > max_batch || ne1 > max_batch || ne10 > max_batch) {
                        वापस false;
                }

                चयन (weights->type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                        // the inputs ztensor is used, and views have none
                        वापस inputs->view_src == nullptr;
                    हाल GGML_TYPE_Q8_0:
                        // the inputs are quantized from their data, so views work as well
                        वापस ctx_dev->has_parmblkformat_1 && inputs->type == GGML_TYPE_F32;
                    शेष:
                        वापस false;
                }
            } अवरोध;

        हाल GGML_OP_FLASH_ATTN_EXT:
            वापस ggml_zdnn_supports_flash_attn_ext(ctx_dev, op);

        शेष:
            वापस false;
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// globals
//

// initialised in ggml_backend_zdnn_reg
अटल ggml_backend_reg    g_ggml_backend_zdnn_reg;
अटल ggml_backend_device g_ggml_backend_zdnn_device;

अटल ggml_backend_zdnn_device_context g_ggml_ctx_dev_main = {
    /* .zdnn_device           = */ 0,
    /* .zdnn_device_ref_count = */ 0,
    /* .has_parmblkformat_0   = */ false,
    /* .has_parmblkformat_1   = */ false,
    /* .max_size              = */ 0,
    /* .name                  = */ "",
};

अटल पूर्णांक ggml_backend_zdnn_device_acq(ggml_backend_zdnn_device_context * ctx) {
    निश्चित(ctx != शून्य);

    अगर (ctx->zdnn_device == 0) {
        ctx->zdnn_device = 1;
    }

    अगर (ctx->zdnn_device >= 1) {
        ctx->has_parmblkformat_0 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0);
        ctx->has_parmblkformat_1 = zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1);
        ctx->max_size = zdnn_get_nnpa_max_dim_idx_आकार();
        म_नकलन(ctx->name, GGML_ZDNN_NAME, माप(ctx->name) - 1);
    }

    ctx->zdnn_device_ref_गिनती++;
    वापस ctx->zdnn_device;
}

अटल व्योम ggml_backend_zdnn_device_rel(ggml_backend_zdnn_device_context * ctx) {
    निश्चित(ctx != शून्य);
    निश्चित(ctx->zdnn_device_ref_गिनती > 0);

    ctx->zdnn_device_ref_गिनती--;
    अगर (ctx->zdnn_device_ref_गिनती == 0) {
        अगर (ctx->zdnn_device >= 0) {
            ctx->zdnn_device = 0;
        }
    }
}

अटल ggml_backend_zdnn_context * ggml_zdnn_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);
    GGML_LOG_INFO("%s: found 1 device\n", __func__);

    #अगर_घोषित STATIC_LIB
    zdnn_init();
    #पूर्ण_अगर

    ggml_backend_zdnn_context * ctx = नया ggml_backend_zdnn_context();
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)dev->context;

    पूर्णांक device = 1;
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, ctx_dev->name);

    ctx->device = device;
    GGML_LOG_INFO("%s: NNPA name: %s\n", __func__, ctx_dev->name);
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_0 = %s\n", __func__, ctx_dev->has_parmblkformat_0 ? "true" : "false");
    GGML_LOG_INFO("%s: NNPA_PARMBLKFORMAT_1 = %s\n", __func__, ctx_dev->has_parmblkformat_1 ? "true" : "false");

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->gf = nullptr;

    वापस ctx;
}

अटल व्योम ggml_zdnn_free(ggml_backend_zdnn_context * ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    क्रम (स्थिर स्वतः & it : ctx->zero_bias) {
        ZDNN_CHECK(zdnn_free_ztensor_बफर(&it.second->ztensor));
    }

    मिटाओ ctx;
}

//
// backend interface
//

अटल व्योम ggml_backend_zdnn_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    क्रम (स्थिर स्वतः & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_बफर * buf = buf_ptr.लो();

        // Free any extra buffer allocated for the tensor. E.g., bias for GGML_OP_MUL_MAT
        अगर (buf->extra != nullptr) मुक्त(buf->extra->तथ्य);
        अगर (buf->ztensor.बफर_आकार > 0) ZDNN_CHECK(zdnn_free_ztensor_बफर(&buf->ztensor));
    }

    अगर (ctx->owned) {
        ggml_aligned_free(ctx->all_तथ्य, ctx->all_आकार);
    }

    मिटाओ ctx;
}

अटल व्योम * ggml_backend_zdnn_बफर_get_base(ggml_backend_बफर_t बफर) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;
    वापस ctx->all_तथ्य;
}

अटल क्रमागत ggml_status ggml_backend_zdnn_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    अगर (tensor->view_src != शून्य) {
        निश्चित(tensor->view_src->बफर->buft == बफर->buft);
        वापस GGML_STATUS_SUCCESS;
    }

    // reject empty tensors to avoid zDNN crash
    अगर (ggml_is_empty(tensor)) {
        वापस GGML_STATUS_SUCCESS;
    }

    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    स्थिर int64_t tsize = ggml_nbytes(tensor);
    पूर्णांक बफर_idx = ctx->n_buffers;

    std::unique_ptr<ggml_backend_zdnn_बफर> zdnn_बफर = std::make_unique<ggml_backend_zdnn_बफर>();
    zdnn_बफर->तथ्य = tensor->तथ्य;
    zdnn_बफर->size = tsize;
    zdnn_बफर->extra = nullptr;
    snprintf(zdnn_बफर->name, GGML_MAX_NAME, "%s", tensor->name);

    ggml_zdnn_init_tensor(zdnn_बफर.लो(), tensor);
    tensor->extra = zdnn_बफर.लो();

    चयन (tensor->op) {
        हाल GGML_OP_MUL_MAT:
            {
                std::unique_ptr<ggml_backend_zdnn_बफर> zdnn_bias_बफर = std::make_unique<ggml_backend_zdnn_बफर>();
                zdnn_bias_बफर->तथ्य = (व्योम *)सुस्मृति(tensor->ne[0], ggml_element_आकार(tensor));
                zdnn_bias_बफर->size = ggml_element_आकार(tensor) * tensor->ne[0];
                snprintf(zdnn_bias_बफर->name, GGML_MAX_NAME, "%.*s (bias)",
                         GGML_MAX_NAME - (पूर्णांक)माप(" (bias)"), tensor->name);

                स्थिर int64_t bias_dim[GGML_MAX_DIMS] = { 1, 1, 1, tensor->ne[0] };
                ggml_zdnn_create_tensor(zdnn_bias_बफर->pre_tfm_desc,
                                        zdnn_bias_बफर->tfm_desc,
                                        zdnn_bias_बफर->ztensor,
                                        tensor, bias_dim, ZDNN_1D);

                ggml_zdnn_load_tensor(zdnn_bias_बफर->ztensor, zdnn_bias_बफर->तथ्य);
                zdnn_बफर->extra = zdnn_bias_बफर.लो();

                ctx->buffers.push_back(std::move(zdnn_bias_बफर));
                ctx->n_buffers++;
            } अवरोध;
        शेष:
            अवरोध;
    }

    ctx->buffers.push_back(std::move(zdnn_बफर));
    ctx->n_buffers++;

    // GGML_LOG_INFO("%s: initialised tensor '%s' in buffer %d, size = %8.2f MiB\n",
    //               __func__, tensor->name, buffer_idx, tsize);

    वापस GGML_STATUS_SUCCESS;

    GGML_UNUSED(बफर_idx);
}

अटल व्योम ggml_backend_zdnn_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    स_रखो((अक्षर *)tensor->तथ्य + offset, value, size);

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_zdnn_बफर_set_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स_नकल((अक्षर *)tensor->तथ्य + offset, तथ्य, size);

    // views share the ztensor of their source, which is now stale, and empty tensors have none
    अगर (tensor->view_src != nullptr) {
        ggml_backend_zdnn_बफर * src_extra = (ggml_backend_zdnn_बफर *)tensor->view_src->extra;
        अगर (src_extra != nullptr && src_extra->ztensor.is_transformed) {
            zdnn_reset_ztensor(&src_extra->ztensor);
        }
        वापस;
    }

    ggml_backend_zdnn_बफर * extra = (ggml_backend_zdnn_बफर *)tensor->extra;
    अगर (extra == nullptr) {
        वापस;
    }

    अगर (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        ggml_zdnn_load_tensor_q(extra, tensor);
        वापस;
    }

    // Fixes the LLAMA_SET_ROWS bug
    // see: https://github.com/ggml-org/llama.cpp/issues/15414
    // only weights are transformed here, others on first use, as not every op reads the ztensor and some data (masks) is not finite
    अगर (tensor->बफर->usage != GGML_BACKEND_बफर_USAGE_WEIGHTS) {
        zdnn_reset_ztensor(&extra->ztensor);
        वापस;
    }
    अगर (extra->ztensor.is_transformed == false) ggml_zdnn_load_tensor(extra->ztensor, tensor->तथ्य);

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_zdnn_बफर_get_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    स_नकल(तथ्य, (स्थिर अक्षर *)tensor->तथ्य + offset, size);

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_zdnn_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    स_रखो(ctx->all_तथ्य, value, ctx->all_आकार);
}

अटल व्योम ggml_backend_zdnn_बफर_reset(ggml_backend_बफर_t बफर) {
    ggml_backend_zdnn_बफर_context * ctx = (ggml_backend_zdnn_बफर_context *)बफर->context;

    क्रम (स्थिर स्वतः & buf_ptr : ctx->buffers) {
        ggml_backend_zdnn_बफर * buf = buf_ptr.लो();

        अगर (buf->extra != nullptr) {
            मुक्त(buf->extra->तथ्य);
        }

        अगर (buf->ztensor.बफर_आकार > 0) {
            ZDNN_CHECK(zdnn_free_ztensor_बफर(&buf->ztensor));
        }
    }

    अगर (ctx->buffers.size() > 1) {
        ctx->buffers.resize(1);
    }

    ctx->n_buffers = 1;
}

अटल ggml_backend_बफर_i ggml_backend_zdnn_बफर_i = {
    /* .free_buffer   = */ ggml_backend_zdnn_बफर_free_बफर,
    /* .get_base      = */ ggml_backend_zdnn_बफर_get_base,
    /* .init_tensor   = */ ggml_backend_zdnn_बफर_init_tensor,
    /* .memset_tensor = */ ggml_backend_zdnn_बफर_memset_tensor,
    /* .set_tensor    = */ ggml_backend_zdnn_बफर_set_tensor,
    /* .get_tensor    = */ ggml_backend_zdnn_बफर_get_tensor,
    /* .set_tensor_2d = */ शून्य,
    /* .get_tensor_2d = */ शून्य,
    /* .cpy_tensor    = */ शून्य,
    /* .clear         = */ ggml_backend_zdnn_बफर_clear,
    /* .reset         = */ ggml_backend_zdnn_बफर_reset,
};

//
// default buffer type
//

अटल स्थिर अक्षर * ggml_backend_zdnn_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस GGML_ZDNN_NAME;

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_t ggml_backend_zdnn_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    ggml_backend_zdnn_बफर_context * ctx = नया ggml_backend_zdnn_बफर_context();

    स्थिर माप_प्रकार आकार_page = sysconf(_SC_PAGESIZE);

    माप_प्रकार आकार_aligned = size;
    अगर ((आकार_aligned % आकार_page) != 0) {
        आकार_aligned += आकार_page - (आकार_aligned % आकार_page);
    }

    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *)buft->device->context;

    GGML_ASSERT(ctx_dev->zdnn_device >= 0);
    पूर्णांक device = ctx_dev->zdnn_device; GGML_UNUSED(device);

    ctx->all_तथ्य  = ggml_aligned_malloc(आकार_aligned);
    ctx->all_आकार  = आकार_aligned;
    ctx->owned     = true;
    ctx->n_buffers = 1;

    अगर (ctx->all_तथ्य != शून्य) {
        std::unique_ptr<ggml_backend_zdnn_बफर> zdnn_बफर = std::make_unique<ggml_backend_zdnn_बफर>();
        zdnn_बफर->तथ्य = ctx->all_तथ्य;
        zdnn_बफर->size = आकार_aligned;
        ctx->buffers.push_back(std::move(zdnn_बफर));
    }

    अगर (आकार_aligned > 0 && (ctx->all_तथ्य == शून्य)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f\n",
                       __func__, आकार_aligned / 1024.0 / 1024.0);
        मिटाओ ctx;
        वापस शून्य;
    }

    वापस ggml_backend_बफर_init(buft, ggml_backend_zdnn_बफर_i, ctx, size);
}

अटल माप_प्रकार ggml_backend_zdnn_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस 256;

    GGML_UNUSED(buft);
}

अटल bool ggml_backend_zdnn_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    /* while it resides in host memory, additional transformation is needed */
    वापस false;

    GGML_UNUSED(buft);
}

ggml_backend_बफर_type_t ggml_backend_zdnn_बफर_type(व्योम) {
    अटल ggml_backend_बफर_type ggml_backend_बफर_type_zdnn = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_zdnn_बफर_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_zdnn_बफर_type_alloc_बफर,
            /* .alloc_buffer_n      = */ शून्य,
            /* .get_alignment       = */ ggml_backend_zdnn_बफर_type_get_alignment,
            /* .get_max_size        = */ शून्य,
            /* .get_alloc_size      = */ शून्य,  // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ शून्य,
            /* .is_host             = */ ggml_backend_zdnn_बफर_type_is_host,
        },
        /* .device  = */ &g_ggml_backend_zdnn_device,
        /* .context = */ शून्य,
    };

    वापस &ggml_backend_बफर_type_zdnn;
}

//
// backend
//

अटल स्थिर अक्षर * ggml_backend_zdnn_name(ggml_backend_t backend) {
    वापस GGML_ZDNN_NAME;

    GGML_UNUSED(backend);
}

अटल व्योम ggml_backend_zdnn_free(ggml_backend_t backend) {
    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;

    ggml_zdnn_free(ctx);
    मुक्त(backend);
}

अटल क्रमागत ggml_status ggml_backend_zdnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    वापस ggml_zdnn_graph_compute(backend, cgraph);
}

अटल ggml_backend_i ggml_backend_zdnn_i = {
    /* .get_name               = */ ggml_backend_zdnn_name,
    /* .free                   = */ ggml_backend_zdnn_free,
    /* .set_tensor_async       = */ शून्य,
    /* .get_tensor_async       = */ शून्य,
    /* .set_tensor_2d_async    = */ शून्य,
    /* .get_tensor_2d_async    = */ शून्य,
    /* .cpy_tensor_async       = */ शून्य,
    /* .synchronize            = */ शून्य,
    /* .graph_plan_create      = */ शून्य,
    /* .graph_plan_free        = */ शून्य,
    /* .graph_plan_update      = */ शून्य,
    /* .graph_plan_compute     = */ शून्य,
    /* .graph_compute          = */ ggml_backend_zdnn_graph_compute,
    /* .event_record           = */ शून्य,
    /* .event_wait             = */ शून्य,
    /* .graph_optimize         = */ शून्य,
};

अटल ggml_guid_t ggml_backend_zdnn_guid(व्योम) {
    अटल अक्षर guid_str[] = "IBM-ZDNN-ACCELER";
    वापस reinterpret_cast<ggml_guid_t>(guid_str);
}

bool ggml_backend_is_zdnn(ggml_backend_t backend) {
    वापस backend != शून्य &&
           ggml_guid_matches(backend->guid, ggml_backend_zdnn_guid());
}

//
// backend device
//

अटल स्थिर अक्षर * ggml_backend_zdnn_device_get_name(ggml_backend_dev_t dev) {
    वापस GGML_ZDNN_NAME;

    GGML_UNUSED(dev);
}

अटल स्थिर अक्षर * ggml_backend_zdnn_device_get_description(ggml_backend_dev_t dev) {
    वापस "IBM Z Neural Network Processing Assist (NNPA)";

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_zdnn_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    *मुक्त  = 0;
    *योग = 0;

    GGML_UNUSED(dev);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_zdnn_device_get_type(ggml_backend_dev_t dev) {
    वापस GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_zdnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zdnn_device_get_name(dev);
    props->description = ggml_backend_zdnn_device_get_description(dev);
    props->type        = ggml_backend_zdnn_device_get_type(dev);
    ggml_backend_zdnn_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ true,
    };
}

अटल ggml_backend_t ggml_backend_zdnn_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    ggml_backend_zdnn_context * ctx = ggml_zdnn_init(dev);
    अगर (ctx == शून्य) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        वापस शून्य;
    }

    ggml_backend_t backend = (ggml_backend *)दो_स्मृति(माप(ggml_backend));
    *backend = {
        /* .guid       = */ ggml_backend_zdnn_guid(),
        /* .iface      = */ ggml_backend_zdnn_i,
        /* .device     = */ dev,
        /* .context    = */ ctx
    };

    वापस backend;

    GGML_UNUSED(params);
}

अटल ggml_backend_बफर_type_t ggml_backend_zdnn_device_get_बफर_type(ggml_backend_dev_t dev) {
    वापस ggml_backend_zdnn_बफर_type();

    GGML_UNUSED(dev);
}

अटल bool ggml_backend_zdnn_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_backend_zdnn_device_context * ctx_dev = (ggml_backend_zdnn_device_context *) dev->context;

    वापस ggml_zdnn_supports_op(ctx_dev, op);
}

अटल bool ggml_backend_zdnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    वापस
        buft->iface.get_name == ggml_backend_zdnn_बफर_type_get_name;

    GGML_UNUSED(dev);
}

अटल ggml_backend_device_i ggml_backend_zdnn_device_i = {
    /* .get_name             = */ ggml_backend_zdnn_device_get_name,
    /* .get_description      = */ ggml_backend_zdnn_device_get_description,
    /* .get_memory           = */ ggml_backend_zdnn_device_get_memory,
    /* .get_type             = */ ggml_backend_zdnn_device_get_type,
    /* .get_props            = */ ggml_backend_zdnn_device_get_props,
    /* .init_backend         = */ ggml_backend_zdnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_zdnn_device_get_बफर_type,
    /* .get_host_buffer_type = */ शून्य,
    /* .buffer_from_host_ptr = */ शून्य,
    /* .supports_op          = */ ggml_backend_zdnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_zdnn_device_supports_buft,
    /* .offload_op           = */ शून्य,
    /* .event_new            = */ शून्य,
    /* .event_free           = */ शून्य,
    /* .event_synchronize    = */ शून्य,
};

//
// backend registry
//

अटल स्थिर अक्षर * ggml_backend_zdnn_reg_get_name(ggml_backend_reg_t reg) {
    वापस GGML_ZDNN_NAME;

    GGML_UNUSED(reg);
}

अटल माप_प्रकार ggml_backend_zdnn_reg_device_गिनती(ggml_backend_reg_t reg) {
    अगर (!zdnn_is_nnpa_installed()) {
        वापस 0;
    }
    वापस 1;

    GGML_UNUSED(reg);
}

अटल ggml_backend_dev_t ggml_backend_zdnn_reg_device_get(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    GGML_ASSERT(सूचकांक == 0);

    वापस &g_ggml_backend_zdnn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(सूचकांक);
}

अटल ggml_backend_feature g_ggml_backend_zdnn_features[] = {
    { "NNPA", zdnn_is_nnpa_installed() ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_0", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) ? "1" : "0" },
    { "NNPA_PARMBLKFORMAT_1", zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) ? "1" : "0" },
    { शून्य, शून्य },
};

अटल ggml_backend_feature * ggml_backend_zdnn_get_features(ggml_backend_reg_t reg) {
    वापस g_ggml_backend_zdnn_features;

    GGML_UNUSED(reg);
}

अटल व्योम ggml_backend_zdnn_set_n_threads(ggml_backend_t backend, पूर्णांक n_threads) {
    GGML_ASSERT(ggml_backend_is_zdnn(backend));

    ggml_backend_zdnn_context * ctx = (ggml_backend_zdnn_context *)backend->context;
    ctx->n_threads = n_threads;
}

अटल व्योम * ggml_backend_zdnn_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    अगर (म_भेद(name, "ggml_backend_get_features") == 0) {
        वापस (व्योम *) ggml_backend_zdnn_get_features;
    }
    अगर (म_भेद(name, "ggml_backend_set_n_threads") == 0) {
        वापस (व्योम *) ggml_backend_zdnn_set_n_threads;
    }

    वापस शून्य;

    GGML_UNUSED(reg);
}

अटल ggml_backend_reg_i ggml_backend_zdnn_reg_i = {
    /* .get_name         = */ ggml_backend_zdnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_zdnn_reg_device_गिनती,
    /* .get_device       = */ ggml_backend_zdnn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_zdnn_get_proc_address
};

अटल व्योम ggml_zdnn_cleanup(व्योम) {
    ggml_backend_zdnn_device_rel(&g_ggml_ctx_dev_main);
}

// TODO: make thread-safe
ggml_backend_reg_t ggml_backend_zdnn_reg(व्योम) {
    ggml_backend_zdnn_device_acq(&g_ggml_ctx_dev_main);

    // register cleanup callback
    निकास_पर(ggml_zdnn_cleanup);

    {
        g_ggml_backend_zdnn_reg = {
            /* .api_version = */ GGML_ZDNN_VERSION,
            /* .iface       = */ ggml_backend_zdnn_reg_i,
            /* .context     = */ शून्य
        };

        g_ggml_backend_zdnn_device = {
            /* .iface       = */ ggml_backend_zdnn_device_i,
            /* .reg         = */ &g_ggml_backend_zdnn_reg,
            /* .context     = */ &g_ggml_ctx_dev_main
        };

        वापस &g_ggml_backend_zdnn_reg;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_zdnn_reg)
