#समावेश "ggml-metal.h"

#समावेश "ggml-impl.h"
#समावेश "ggml-backend-impl.h"

#समावेश "ggml-metal-device.h"
#समावेश "ggml-metal-fusion.h"
#समावेश "ggml-metal-context.h"
#समावेश "ggml-metal-ops.h"
#समावेश "ggml-metal-tuning.h"

#समावेश <mutex>
#समावेश <string>

#घोषणा GGML_METAL_NAME "MTL"
#घोषणा GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
अटल पूर्णांक g_devices = 1;

// forward declaration
अटल bool ggml_backend_बफर_is_metal(ggml_backend_बफर_t बफर);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

अटल व्योम ggml_backend_metal_बफर_shared_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_free(ctx);
}

अटल व्योम * ggml_backend_metal_बफर_shared_get_base(ggml_backend_बफर_t बफर) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(ggml_metal_बफर_is_shared(ctx));

    वापस ggml_metal_बफर_get_base(ctx);
}

अटल व्योम ggml_backend_metal_बफर_shared_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_memset_tensor(ctx, tensor, value, offset, size);
}

अटल व्योम ggml_backend_metal_बफर_shared_set_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_set_tensor(ctx, tensor, तथ्य, offset, size);
}

अटल व्योम ggml_backend_metal_बफर_shared_get_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_get_tensor(ctx, tensor, तथ्य, offset, size);
}

अटल bool ggml_backend_metal_बफर_shared_cpy_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(ggml_metal_बफर_is_shared(ctx));

    अगर (!ggml_backend_बफर_is_metal(src->बफर)) {
        वापस false;
    }

    वापस ggml_metal_बफर_cpy_tensor(ctx, src, dst);
}

अटल व्योम ggml_backend_metal_बफर_shared_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_clear(ctx, value);
}

अटल ggml_backend_बफर_i ggml_backend_metal_बफर_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_बफर_shared_free_बफर,
    /* .get_base      = */ ggml_backend_metal_बफर_shared_get_base,
    /* .init_tensor   = */ शून्य,
    /* .memset_tensor = */ ggml_backend_metal_बफर_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_बफर_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_बफर_shared_get_tensor,
    /* .set_tensor_2d = */ शून्य,
    /* .get_tensor_2d = */ शून्य,
    /* .cpy_tensor    = */ ggml_backend_metal_बफर_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_बफर_shared_clear,
    /* .reset         = */ शून्य,
};

// private buffer

अटल व्योम ggml_backend_metal_बफर_private_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(!ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_free(ctx);
}

अटल व्योम * ggml_backend_metal_बफर_private_get_base(ggml_backend_बफर_t बफर) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(!ggml_metal_बफर_is_shared(ctx));

    वापस ggml_metal_बफर_get_base(ctx);
}

अटल व्योम ggml_backend_metal_बफर_private_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(!ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_memset_tensor(ctx, tensor, value, offset, size);
}

अटल व्योम ggml_backend_metal_बफर_private_set_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(!ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_set_tensor(ctx, tensor, तथ्य, offset, size);
}

अटल व्योम ggml_backend_metal_बफर_private_get_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(!ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_get_tensor(ctx, tensor, तथ्य, offset, size);
}

अटल bool ggml_backend_metal_बफर_private_cpy_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(!ggml_metal_बफर_is_shared(ctx));

    अगर (!ggml_backend_बफर_is_metal(src->बफर)) {
        वापस false;
    }

    वापस ggml_metal_बफर_cpy_tensor(ctx, src, dst);
}

अटल व्योम ggml_backend_metal_बफर_private_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_metal_बफर_t ctx = (ggml_metal_बफर_t)बफर->context;

    GGML_ASSERT(!ggml_metal_बफर_is_shared(ctx));

    ggml_metal_बफर_clear(ctx, value);
}

अटल ggml_backend_बफर_i ggml_backend_metal_बफर_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_बफर_private_free_बफर,
    /* .get_base      = */ ggml_backend_metal_बफर_private_get_base,
    /* .init_tensor   = */ शून्य,
    /* .memset_tensor = */ ggml_backend_metal_बफर_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_बफर_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_बफर_private_get_tensor,
    /* .set_tensor_2d = */ शून्य,
    /* .get_tensor_2d = */ शून्य,
    /* .cpy_tensor    = */ ggml_backend_metal_बफर_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_बफर_private_clear,
    /* .reset         = */ शून्य,
};

अटल bool ggml_backend_बफर_is_metal(ggml_backend_बफर_t बफर) {
    वापस बफर->iface.free_बफर == ggml_backend_metal_बफर_shared_free_बफर ||
           बफर->iface.free_बफर == ggml_backend_metal_बफर_private_free_बफर;
}

//
// buffer types
//

काष्ठा ggml_backend_metal_बफर_type {
    पूर्णांक device;
    std::string name;
};

काष्ठा ggml_backend_metal_बफर_type_deleter {
    व्योम चालक()(ggml_backend_metal_बफर_type * ctx) स्थिर {
        मिटाओ ctx;
    }
};

प्रकार std::unique_ptr<ggml_backend_metal_बफर_type, ggml_backend_metal_बफर_type_deleter> ggml_backend_metal_बफर_type_ptr;

// common method for allocating shread or private Metal buffers
अटल ggml_backend_बफर_t ggml_backend_metal_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size, bool shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_बफर_t res = ggml_metal_बफर_init(ctx_dev, size, shared);

    अगर (res == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        वापस शून्य;
    }

    ggml_backend_बफर_i buf_i = ggml_metal_बफर_is_shared(res)
        ? ggml_backend_metal_बफर_shared_i
        : ggml_backend_metal_बफर_private_i;

    वापस ggml_backend_बफर_init(buft, buf_i, res, size);
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    माप_प्रकार res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    चयन (tensor->op) {
        हाल GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } अवरोध;
        हाल GGML_OP_CUMSUM:
        हाल GGML_OP_ARGSORT:
            {
                res *= 2;
            } अवरोध;
        हाल GGML_OP_शीर्ष_K:
            {
                res = 2*माप(int32_t)*ggml_nelements(tensor->src[0]);
            } अवरोध;
        हाल GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } अवरोध;
        शेष:
            अवरोध;
    }

    वापस res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

अटल स्थिर अक्षर * ggml_backend_metal_बफर_type_shared_get_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_metal_बफर_type * ctx = (ggml_backend_metal_बफर_type *)buft->context;

    वापस ctx->name.c_str();
}

अटल ggml_backend_बफर_t ggml_backend_metal_बफर_type_shared_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    वापस ggml_backend_metal_बफर_type_alloc_बफर(buft, size, true);
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_shared_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस 32;

    GGML_UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_shared_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    वापस ggml_metal_device_get_props(ctx_dev)->max_बफर_आकार;
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_shared_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    वापस ggml_backend_metal_बफर_type_get_alloc_आकार(buft, tensor);
}

अटल bool ggml_backend_metal_बफर_type_shared_is_host(ggml_backend_बफर_type_t buft) {
    वापस false;

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_type_t ggml_backend_metal_बफर_type_shared(पूर्णांक device) {
    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    अटल std::vector<ggml_backend_बफर_type> bufts;
    अटल std::vector<ggml_backend_metal_बफर_type_ptr> ctxs;

    अटल bool initialized = false;
    अगर (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        क्रम (पूर्णांक i = 0; i < g_devices; ++i) {
            ggml_backend_metal_बफर_type * raw_ctx =
                नया ggml_backend_metal_बफर_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_बफर_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_बफर_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_बफर_type_shared_alloc_बफर,
                    /* .alloc_buffer_n      = */ शून्य,
                    /* .get_alignment       = */ ggml_backend_metal_बफर_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_बफर_type_shared_get_max_आकार,
                    /* .get_alloc_size      = */ ggml_backend_metal_बफर_type_shared_get_alloc_आकार,
                    /* .get_alloc_size_n    = */ शून्य,
                    /* .is_host             = */ ggml_backend_metal_बफर_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    वापस &bufts[device];
}

// default (private) buffer type

अटल स्थिर अक्षर * ggml_backend_metal_बफर_type_private_get_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_metal_बफर_type * ctx = (ggml_backend_metal_बफर_type *)buft->context;

    वापस ctx->name.c_str();
}

अटल ggml_backend_बफर_t ggml_backend_metal_बफर_type_private_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    वापस ggml_backend_metal_बफर_type_alloc_बफर(buft, size, false);
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_private_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस 32;

    GGML_UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_private_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    वापस ggml_metal_device_get_props(ctx_dev)->max_बफर_आकार;
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_private_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    वापस ggml_backend_metal_बफर_type_get_alloc_आकार(buft, tensor);
}

अटल bool ggml_backend_metal_बफर_type_private_is_host(ggml_backend_बफर_type_t buft) {
    वापस false;

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_type_t ggml_backend_metal_बफर_type_private(पूर्णांक device) {
    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    अटल std::vector<ggml_backend_बफर_type> bufts;
    अटल std::vector<ggml_backend_metal_बफर_type_ptr> ctxs;

    अटल bool initialized = false;
    अगर (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        क्रम (पूर्णांक i = 0; i < g_devices; ++i) {
            ggml_backend_metal_बफर_type * raw_ctx = नया ggml_backend_metal_बफर_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_बफर_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_बफर_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_बफर_type_private_alloc_बफर,
                    /* .alloc_buffer_n      = */ शून्य,
                    /* .get_alignment       = */ ggml_backend_metal_बफर_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_बफर_type_private_get_max_आकार,
                    /* .get_alloc_size      = */ ggml_backend_metal_बफर_type_private_get_alloc_आकार,
                    /* .get_alloc_size_n    = */ शून्य,
                    /* .is_host             = */ ggml_backend_metal_बफर_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    वापस &bufts[device];
}

// mapped buffer type

अटल स्थिर अक्षर * ggml_backend_metal_बफर_type_mapped_get_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_metal_बफर_type * ctx = (ggml_backend_metal_बफर_type *)buft->context;

    वापस ctx->name.c_str();
}

अटल ggml_backend_बफर_t ggml_backend_metal_बफर_type_mapped_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    // for mapped buffers, prefer shared memory
    वापस ggml_backend_metal_बफर_type_alloc_बफर(buft, size, true);
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_mapped_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस 32;

    GGML_UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_mapped_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    वापस ggml_metal_device_get_props(ctx_dev)->max_बफर_आकार;
}

अटल माप_प्रकार ggml_backend_metal_बफर_type_mapped_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    वापस ggml_backend_metal_बफर_type_get_alloc_आकार(buft, tensor);
}

अटल bool ggml_backend_metal_बफर_type_mapped_is_host(ggml_backend_बफर_type_t buft) {
    वापस false;

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_type_t ggml_backend_metal_बफर_type_mapped(पूर्णांक device) {
    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    अटल std::vector<ggml_backend_बफर_type> bufts;
    अटल std::vector<ggml_backend_metal_बफर_type_ptr> ctxs;

    अटल bool initialized = false;
    अगर (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        क्रम (पूर्णांक i = 0; i < g_devices; ++i) {
            ggml_backend_metal_बफर_type * raw_ctx = नया ggml_backend_metal_बफर_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_बफर_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_बफर_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_बफर_type_mapped_alloc_बफर,
                    /* .alloc_buffer_n      = */ शून्य,
                    /* .get_alignment       = */ ggml_backend_metal_बफर_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_बफर_type_mapped_get_max_आकार,
                    /* .get_alloc_size      = */ ggml_backend_metal_बफर_type_mapped_get_alloc_आकार,
                    /* .get_alloc_size_n    = */ शून्य,
                    /* .is_host             = */ ggml_backend_metal_बफर_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    वापस &bufts[device];
}

// backend

अटल स्थिर अक्षर * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    वापस ggml_metal_get_name(ctx);
}

अटल व्योम ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    मुक्त(backend);
}

अटल व्योम ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

अटल व्योम ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, तथ्य, offset, size);
}

अटल व्योम ggml_backend_metal_get_tensor_async(ggml_backend_t backend, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, तथ्य, offset, size);
}

अटल bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    अगर (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        वापस false;
    }

    अगर (!ggml_backend_बफर_is_metal(src->बफर) || !ggml_backend_बफर_is_metal(dst->बफर)) {
        वापस false;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    वापस ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

अटल क्रमागत ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    वापस ggml_metal_graph_compute(ctx, cgraph);
}

अटल व्योम ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

अटल व्योम ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

अटल व्योम ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_तथ्य, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

अटल व्योम ggml_backend_metal_set_n_cb(ggml_backend_t backend, पूर्णांक n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

अटल ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

अटल ggml_guid_t ggml_backend_metal_guid(व्योम) {
    अटल ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    वापस &guid;
}

ggml_backend_t ggml_backend_metal_init(व्योम) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    अगर (ctx == शून्य) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        वापस शून्य;
    }

    ggml_backend_t backend = (ggml_backend_t) दो_स्मृति(माप(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    वापस backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

व्योम ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, व्योम * user_तथ्य) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_तथ्य);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, पूर्णांक family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    वापस ggml_metal_supports_family(ctx, family);
}

व्योम ggml_backend_metal_capture_अग्र_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_अग्र_compute(ctx);
}

// backend device

अटल स्थिर अक्षर * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    स्थिर ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    वापस props_dev->name;
}

अटल स्थिर अक्षर * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    वापस ggml_metal_device_get_props(ctx_dev)->desc;
}

अटल व्योम ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, मुक्त, योग);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    वापस GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_योग);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ true,
        /* .mmap_support         = */ true,
    };
}

अटल ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    अगर (ctx == शून्य) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        वापस शून्य;
    }

    ggml_backend_t backend = (ggml_backend_t) दो_स्मृति(माप(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    वापस backend;

    GGML_UNUSED(params);
}

अटल ggml_backend_बफर_type_t ggml_backend_metal_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    स्थिर ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    वापस props_dev->use_shared_buffers ? ggml_backend_metal_बफर_type_shared(props_dev->device) : ggml_backend_metal_बफर_type_private(props_dev->device);
}

अटल ggml_backend_बफर_t ggml_backend_metal_device_बफर_mapped(ggml_backend_dev_t dev, व्योम * ptr, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_बफर_t res = ggml_metal_बफर_map(ctx_dev, ptr, size, max_tensor_आकार);

    स्थिर ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    वापस ggml_backend_बफर_init(ggml_backend_metal_बफर_type_mapped(props_dev->device), ggml_backend_metal_बफर_shared_i, res, size);
}

अटल bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    वापस ggml_metal_device_supports_op(ctx_dev, op);
}

अटल bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    वापस
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_बफर_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_बफर_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_बफर_type_mapped_get_name);

    GGML_UNUSED(dev);
}

अटल int64_t get_op_batch_आकार(स्थिर ggml_tensor * op) {
    चयन (op->op) {
        हाल GGML_OP_MUL_MAT:
            वापस op->ne[1];
        हाल GGML_OP_MUL_MAT_ID:
            वापस op->ne[2];
        शेष:
            वापस ggml_nrows(op);
    }
}

अटल bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    वापस (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_आकार(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_आकार;
}

अटल ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = नया ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    वापस ev;
}

अटल व्योम ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    मिटाओ event;
}

अटल व्योम ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

अटल ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_बफर_type,
    /* .get_host_buffer_type = */ शून्य,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_बफर_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

काष्ठा ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

प्रकार काष्ठा ggml_backend_metal_reg * ggml_backend_metal_reg_t;

अटल ggml_backend_metal_reg_t ggml_backend_metal_reg_init(व्योम) {
    ggml_backend_metal_reg_t ctx = नया काष्ठा ggml_backend_metal_reg;

    वापस ctx;
}

अटल व्योम ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    मिटाओ ctx;
}

काष्ठा ggml_backend_metal_reg_deleter {
    व्योम चालक()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

प्रकार std::unique_ptr<काष्ठा ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

अटल स्थिर अक्षर * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    वापस GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

अटल माप_प्रकार ggml_backend_metal_reg_device_गिनती(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    वापस ctx->devices.size();
}

अटल ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(सूचकांक < ctx->devices.size());
    वापस ctx->devices[सूचकांक];
}

अटल ggml_backend_feature g_ggml_backend_metal_features[] = {
#अगर defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#पूर्ण_अगर
    { शून्य, शून्य },
};

अटल ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    वापस g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
अटल व्योम ggml_backend_metal_tuning_set_fa_vec_override(पूर्णांक Q, पूर्णांक NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

अटल व्योम ggml_backend_metal_tuning_clear_fa_vec_override(व्योम) {
    ggml_metal_tuning::fa_vec_clear_override();
}

अटल पूर्णांक ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    वापस ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

अटल पूर्णांक ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    वापस ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

अटल पूर्णांक ggml_backend_metal_tuning_fa_vec_baseline_ne(पूर्णांक dk, पूर्णांक dv) {
    वापस ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

अटल स्थिर अक्षर * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    वापस ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
प्रकार व्योम * ggml_backend_fusion_t;

अटल ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    वापस ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

अटल व्योम ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((काष्ठा ggml_metal_fusion_info *) finfo);
}

अटल व्योम ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((काष्ठा ggml_metal_fusion_info *) finfo);
}

अटल पूर्णांक ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, स्थिर अक्षर ** labels, uint64_t * counts, पूर्णांक n) {
    वापस ggml_metal_fusion_info_stats_get((काष्ठा ggml_metal_fusion_info *) finfo, labels, counts, n);
}

अटल व्योम ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, bool enabled) {
    ggml_metal_fusion_info_set_enabled((काष्ठा ggml_metal_fusion_info *) finfo, enabled);
}

अटल व्योम * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    अगर (म_भेद(name, "ggml_backend_get_features") == 0) {
        वापस (व्योम *)ggml_backend_metal_get_features;
    }
    अगर (म_भेद(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        वापस (व्योम *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    अगर (म_भेद(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        वापस (व्योम *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    अगर (म_भेद(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        वापस (व्योम *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    अगर (म_भेद(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        वापस (व्योम *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    अगर (म_भेद(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        वापस (व्योम *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    अगर (म_भेद(name, "ggml_backend_metal_tuning_device_token") == 0) {
        वापस (व्योम *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    अगर (म_भेद(name, "ggml_backend_fusion_get") == 0) {
        वापस (व्योम *)ggml_backend_metal_fusion_get;
    }
    अगर (म_भेद(name, "ggml_backend_fusion_stats_init") == 0) {
        वापस (व्योम *)ggml_backend_metal_fusion_stats_init;
    }
    अगर (म_भेद(name, "ggml_backend_fusion_stats_reset") == 0) {
        वापस (व्योम *)ggml_backend_metal_fusion_stats_reset;
    }
    अगर (म_भेद(name, "ggml_backend_fusion_stats_get") == 0) {
        वापस (व्योम *)ggml_backend_metal_fusion_stats_get;
    }
    अगर (म_भेद(name, "ggml_backend_fusion_set_enabled") == 0) {
        वापस (व्योम *)ggml_backend_metal_fusion_set_enabled;
    }

    वापस शून्य;

    GGML_UNUSED(reg);
}

अटल ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_गिनती,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

अटल ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, पूर्णांक device) {
    वापस नया ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

अटल व्योम ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    मिटाओ dev;
}

काष्ठा ggml_backend_device_deleter {
    व्योम चालक()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

प्रकार std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(व्योम) {
    अटल ggml_backend_reg reg;
    अटल bool initialized = false;

    {
        अटल std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        स्थिर अक्षर * env = दो_पर्या("GGML_METAL_DEVICES");
        अगर (env) {
            g_devices = म_से_प(env);
        }

        अटल std::vector<ggml_backend_device_ptr> devs;

        अगर (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true);

            अटल ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            क्रम (पूर्णांक i = 0; i < g_devices; ++i) {
                स्वतः * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.लो(),
            };
        }

        initialized = true;
    }

    वापस &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
