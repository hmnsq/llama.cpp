#अन्तर्भाव "ggml-metal.h"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव "ggml-metal-device.h"
#अन्तर्भाव "ggml-metal-fusion.h"
#अन्तर्भाव "ggml-metal-context.h"
#अन्तर्भाव "ggml-metal-ops.h"
#अन्तर्भाव "ggml-metal-tuning.h"

#अन्तर्भाव <mutex>
#अन्तर्भाव <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
स्थिर पूर्णाङ्क g_devices = 1;

// forward declaration
स्थिर तार्किक ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

स्थिर शून्य ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

स्थिर शून्य * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    फर्काऊ ggml_metal_buffer_get_base(ctx);
}

स्थिर शून्य ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

स्थिर शून्य ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, अचर शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

स्थिर शून्य ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, अचर ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

स्थिर तार्किक ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, अचर ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    यदि (!ggml_backend_buffer_is_metal(src->buffer)) {
        फर्काऊ असत्य;
    }

    फर्काऊ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

स्थिर शून्य ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

स्थिर ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ रिक्त,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ रिक्त,
    /* .get_tensor_2d = */ रिक्त,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ रिक्त,
};

// private buffer

स्थिर शून्य ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

स्थिर शून्य * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    फर्काऊ ggml_metal_buffer_get_base(ctx);
}

स्थिर शून्य ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

स्थिर शून्य ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, अचर शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

स्थिर शून्य ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, अचर ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

स्थिर तार्किक ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, अचर ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    यदि (!ggml_backend_buffer_is_metal(src->buffer)) {
        फर्काऊ असत्य;
    }

    फर्काऊ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

स्थिर शून्य ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

स्थिर ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ रिक्त,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ रिक्त,
    /* .get_tensor_2d = */ रिक्त,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ रिक्त,
};

स्थिर तार्किक ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    फर्काऊ buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

संरचना ggml_backend_metal_buffer_type {
    पूर्णाङ्क device;
    std::string name;
};

संरचना ggml_backend_metal_buffer_type_deleter {
    शून्य चालक()(ggml_backend_metal_buffer_type * ctx) अचर {
        delete ctx;
    }
};

प्रकारनिर्देश std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
स्थिर ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size, तार्किक shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    यदि (res == रिक्त) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        फर्काऊ रिक्त;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    फर्काऊ ggml_backend_buffer_init(buft, buf_i, res, size);
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, अचर ggml_tensor * tensor) {
    माप_प्रकार res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    विकल्प (tensor->op) {
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } रोक;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } रोक;
        पक्ष GGML_OP_CUMSUM:
        पक्ष GGML_OP_ARGSORT:
            {
                res *= 2;
            } रोक;
        पक्ष GGML_OP_शीर्ष_K:
            {
                res = 2*परिमाण(int32_t)*ggml_nelements(tensor->src[0]);
            } रोक;
        पक्ष GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } रोक;
        स्वभाव:
            रोक;
    }

    फर्काऊ res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

स्थिर अचर वर्ण * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    फर्काऊ ctx->name.c_str();
}

स्थिर ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    फर्काऊ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, सत्य);
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    फर्काऊ 32;

    GGML_UNUSED(buft);
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    फर्काऊ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, अचर ggml_tensor * tensor) {
    फर्काऊ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

स्थिर तार्किक ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    फर्काऊ असत्य;

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(पूर्णाङ्क device) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    स्थिर std::vector<ggml_backend_buffer_type> bufts;
    स्थिर std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    स्थिर तार्किक initialized = असत्य;
    यदि (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        लागि (पूर्णाङ्क i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                नूतन ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ रिक्त,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ रिक्त,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = सत्य;
    }

    फर्काऊ &bufts[device];
}

// default (private) buffer type

स्थिर अचर वर्ण * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    फर्काऊ ctx->name.c_str();
}

स्थिर ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    फर्काऊ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, असत्य);
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    फर्काऊ 32;

    GGML_UNUSED(buft);
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    फर्काऊ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, अचर ggml_tensor * tensor) {
    फर्काऊ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

स्थिर तार्किक ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    फर्काऊ असत्य;

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(पूर्णाङ्क device) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    स्थिर std::vector<ggml_backend_buffer_type> bufts;
    स्थिर std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    स्थिर तार्किक initialized = असत्य;
    यदि (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        लागि (पूर्णाङ्क i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = नूतन ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ रिक्त,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ रिक्त,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = सत्य;
    }

    फर्काऊ &bufts[device];
}

// mapped buffer type

स्थिर अचर वर्ण * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    फर्काऊ ctx->name.c_str();
}

स्थिर ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    // for mapped buffers, prefer shared memory
    फर्काऊ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, सत्य);
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    फर्काऊ 32;

    GGML_UNUSED(buft);
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    फर्काऊ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

स्थिर माप_प्रकार ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, अचर ggml_tensor * tensor) {
    फर्काऊ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

स्थिर तार्किक ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    फर्काऊ असत्य;

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(पूर्णाङ्क device) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    स्थिर std::vector<ggml_backend_buffer_type> bufts;
    स्थिर std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    स्थिर तार्किक initialized = असत्य;
    यदि (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        लागि (पूर्णाङ्क i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = नूतन ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .alloc_buffer_n      = */ रिक्त,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ रिक्त,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = सत्य;
    }

    फर्काऊ &bufts[device];
}

// backend

स्थिर अचर वर्ण * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    फर्काऊ ggml_metal_get_name(ctx);
}

स्थिर शून्य ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    मुक्त(backend);
}

स्थिर शून्य ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

स्थिर शून्य ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, अचर शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

स्थिर शून्य ggml_backend_metal_get_tensor_async(ggml_backend_t backend, अचर ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

स्थिर तार्किक ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, अचर ggml_tensor * src, ggml_tensor * dst) {
    यदि (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        फर्काऊ असत्य;
    }

    यदि (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        फर्काऊ असत्य;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    फर्काऊ ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

स्थिर गणना ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    फर्काऊ ggml_metal_graph_compute(ctx, cgraph);
}

स्थिर शून्य ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

स्थिर शून्य ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

स्थिर शून्य ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

स्थिर शून्य ggml_backend_metal_set_n_cb(ggml_backend_t backend, पूर्णाङ्क n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

स्थिर ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

स्थिर ggml_guid_t ggml_backend_metal_guid(शून्य) {
    स्थिर ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    फर्काऊ &guid;
}

ggml_backend_t ggml_backend_metal_init(शून्य) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    यदि (ctx == रिक्त) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        फर्काऊ रिक्त;
    }

    ggml_backend_t backend = (ggml_backend_t) दो_स्मृति(परिमाण(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    फर्काऊ backend;
}

तार्किक ggml_backend_is_metal(ggml_backend_t backend) {
    फर्काऊ backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

शून्य ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, शून्य * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

तार्किक ggml_backend_metal_supports_family(ggml_backend_t backend, पूर्णाङ्क family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    फर्काऊ ggml_metal_supports_family(ctx, family);
}

शून्य ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

स्थिर अचर वर्ण * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    अचर ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    फर्काऊ props_dev->name;
}

स्थिर अचर वर्ण * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    फर्काऊ ggml_metal_device_get_props(ctx_dev)->desc;
}

स्थिर शून्य ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, मुक्त, total);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    फर्काऊ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ सत्य,
        /* .host_buffer          = */ असत्य,
        /* .buffer_from_host_ptr = */ सत्य,
        /* .events               = */ सत्य,
        /* .mmap_support         = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, अचर वर्ण * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    यदि (ctx == रिक्त) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        फर्काऊ रिक्त;
    }

    ggml_backend_t backend = (ggml_backend_t) दो_स्मृति(परिमाण(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    फर्काऊ backend;

    GGML_UNUSED(params);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    अचर ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    फर्काऊ props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

स्थिर ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, शून्य * ptr, माप_प्रकार size, माप_प्रकार max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    अचर ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    फर्काऊ ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

स्थिर तार्किक ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, अचर ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    फर्काऊ ggml_metal_device_supports_op(ctx_dev, op);
}

स्थिर तार्किक ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    फर्काऊ
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

स्थिर int64_t get_op_batch_size(अचर ggml_tensor * op) {
    विकल्प (op->op) {
        पक्ष GGML_OP_MUL_MAT:
            फर्काऊ op->ne[1];
        पक्ष GGML_OP_MUL_MAT_ID:
            फर्काऊ op->ne[2];
        स्वभाव:
            फर्काऊ ggml_nrows(op);
    }
}

स्थिर तार्किक ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, अचर ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    फर्काऊ (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

स्थिर ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = नूतन ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    फर्काऊ ev;
}

स्थिर शून्य ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

स्थिर शून्य ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

स्थिर ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

संरचना ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

प्रकारनिर्देश संरचना ggml_backend_metal_reg * ggml_backend_metal_reg_t;

स्थिर ggml_backend_metal_reg_t ggml_backend_metal_reg_init(शून्य) {
    ggml_backend_metal_reg_t ctx = नूतन संरचना ggml_backend_metal_reg;

    फर्काऊ ctx;
}

स्थिर शून्य ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

संरचना ggml_backend_metal_reg_deleter {
    शून्य चालक()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

प्रकारनिर्देश std::unique_ptr<संरचना ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

स्थिर अचर वर्ण * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    फर्काऊ GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

स्थिर माप_प्रकार ggml_backend_metal_reg_device_गन्ती(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    फर्काऊ ctx->devices.size();
}

स्थिर ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, माप_प्रकार index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    फर्काऊ ctx->devices[index];
}

स्थिर ggml_backend_feature g_ggml_backend_metal_features[] = {
#यदि defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { रिक्त, रिक्त },
};

स्थिर ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    फर्काऊ g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
स्थिर शून्य ggml_backend_metal_tuning_set_fa_vec_override(पूर्णाङ्क Q, पूर्णाङ्क NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

स्थिर शून्य ggml_backend_metal_tuning_clear_fa_vec_override(शून्य) {
    ggml_metal_tuning::fa_vec_clear_override();
}

स्थिर पूर्णाङ्क ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    फर्काऊ ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

स्थिर पूर्णाङ्क ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    फर्काऊ ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

स्थिर पूर्णाङ्क ggml_backend_metal_tuning_fa_vec_baseline_ne(पूर्णाङ्क dk, पूर्णाङ्क dv) {
    फर्काऊ ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

स्थिर अचर वर्ण * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    फर्काऊ ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
प्रकारनिर्देश शून्य * ggml_backend_fusion_t;

स्थिर ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    फर्काऊ ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

स्थिर शून्य ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((संरचना ggml_metal_fusion_info *) finfo);
}

स्थिर शून्य ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((संरचना ggml_metal_fusion_info *) finfo);
}

स्थिर पूर्णाङ्क ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, अचर वर्ण ** labels, uint64_t * counts, पूर्णाङ्क n) {
    फर्काऊ ggml_metal_fusion_info_stats_get((संरचना ggml_metal_fusion_info *) finfo, labels, counts, n);
}

स्थिर शून्य ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, तार्किक enabled) {
    ggml_metal_fusion_info_set_enabled((संरचना ggml_metal_fusion_info *) finfo, enabled);
}

स्थिर शून्य * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, अचर वर्ण * name) {
    यदि (म_भेद(name, "ggml_backend_get_features") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_get_features;
    }
    यदि (म_भेद(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    यदि (म_भेद(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    यदि (म_भेद(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    यदि (म_भेद(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    यदि (म_भेद(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    यदि (म_भेद(name, "ggml_backend_metal_tuning_device_token") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    यदि (म_भेद(name, "ggml_backend_fusion_get") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_fusion_get;
    }
    यदि (म_भेद(name, "ggml_backend_fusion_stats_init") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_fusion_stats_init;
    }
    यदि (म_भेद(name, "ggml_backend_fusion_stats_reset") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_fusion_stats_reset;
    }
    यदि (म_भेद(name, "ggml_backend_fusion_stats_get") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_fusion_stats_get;
    }
    यदि (म_भेद(name, "ggml_backend_fusion_set_enabled") == 0) {
        फर्काऊ (शून्य *)ggml_backend_metal_fusion_set_enabled;
    }

    फर्काऊ रिक्त;

    GGML_UNUSED(reg);
}

स्थिर ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_गन्ती,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

स्थिर ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, पूर्णाङ्क device) {
    फर्काऊ नूतन ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

स्थिर शून्य ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

संरचना ggml_backend_device_deleter {
    शून्य चालक()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

प्रकारनिर्देश std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(शून्य) {
    स्थिर ggml_backend_reg reg;
    स्थिर तार्किक initialized = असत्य;

    {
        स्थिर std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        अचर वर्ण * env = दो_पर्या("GGML_METAL_DEVICES");
        यदि (env) {
            g_devices = म_से_प(env);
        }

        स्थिर std::vector<ggml_backend_device_ptr> devs;

        यदि (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", सत्य);

            स्थिर ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            लागि (पूर्णाङ्क i = 0; i < g_devices; ++i) {
                स्वतः * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = सत्य;
    }

    फर्काऊ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
