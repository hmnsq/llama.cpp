#ଅନ୍ତର୍ଭାଵ "ggml-metal.h"

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଅନ୍ତର୍ଭାଵ "ggml-metal-device.h"
#ଅନ୍ତର୍ଭାଵ "ggml-metal-fusion.h"
#ଅନ୍ତର୍ଭାଵ "ggml-metal-context.h"
#ଅନ୍ତର୍ଭାଵ "ggml-metal-ops.h"
#ଅନ୍ତର୍ଭାଵ "ggml-metal-tuning.h"

#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_devices = 1;

// forward declaration
ସ୍ଥିର ତାର୍କିକ ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ଫେରନ୍ତୁ ggml_metal_buffer_get_base(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ଯଦି (!ggml_backend_buffer_is_metal(src->buffer)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

ସ୍ଥିର ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ ଖାଲି,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ ଖାଲି,
    /* .get_tensor_2d = */ ଖାଲି,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ ଖାଲି,
};

// private buffer

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ଫେରନ୍ତୁ ggml_metal_buffer_get_base(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ଯଦି (!ggml_backend_buffer_is_metal(src->buffer)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

ସ୍ଥିର ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ ଖାଲି,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ ଖାଲି,
    /* .get_tensor_2d = */ ଖାଲି,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ ଖାଲି,
};

ସ୍ଥିର ତାର୍କିକ ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    ଫେରନ୍ତୁ buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

ସଂରଚନା ggml_backend_metal_buffer_type {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device;
    std::string name;
};

ସଂରଚନା ggml_backend_metal_buffer_type_deleter {
    ଶୂନ୍ୟ operator()(ggml_backend_metal_buffer_type * ctx) ଅଚଳ {
        delete ctx;
    }
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
ସ୍ଥିର ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, ତାର୍କିକ shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    ଯଦି (res == ଖାଲି) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    ଫେରନ୍ତୁ ggml_backend_buffer_init(buft, buf_i, res, size);
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    ବାଛ (tensor->op) {
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CUMSUM:
        ସ୍ଥିତି GGML_OP_ARGSORT:
            {
                res *= 2;
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K:
            {
                res = 2*ଆକାର(int32_t)*ggml_nelements(tensor->src[0]);
            } ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }

    ଫେରନ୍ତୁ res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    ଫେରନ୍ତୁ ctx->name.c_str();
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ଫେରନ୍ତୁ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, ସତ୍ୟ);
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ 32;

    GGML_UNUSED(buft);
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ଫେରନ୍ତୁ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ମିଥ୍ୟା;

    GGML_UNUSED(buft);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ସ୍ଥିର std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ସ୍ଥିର std::vector<ggml_backend_buffer_type> bufts;
    ସ୍ଥିର std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;
    ଯଦି (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                ନୂତନ ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ ଖାଲି,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ ଖାଲି,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ &bufts[device];
}

// default (private) buffer type

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    ଫେରନ୍ତୁ ctx->name.c_str();
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ଫେରନ୍ତୁ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, ମିଥ୍ୟା);
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ 32;

    GGML_UNUSED(buft);
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ଫେରନ୍ତୁ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ମିଥ୍ୟା;

    GGML_UNUSED(buft);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ସ୍ଥିର std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ସ୍ଥିର std::vector<ggml_backend_buffer_type> bufts;
    ସ୍ଥିର std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;
    ଯଦି (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = ନୂତନ ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ ଖାଲି,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ ଖାଲି,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ &bufts[device];
}

// mapped buffer type

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    ଫେରନ୍ତୁ ctx->name.c_str();
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    ଫେରନ୍ତୁ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, ସତ୍ୟ);
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ 32;

    GGML_UNUSED(buft);
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ଫେରନ୍ତୁ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ସ୍ଥିର size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    ଫେରନ୍ତୁ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ମିଥ୍ୟା;

    GGML_UNUSED(buft);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ସ୍ଥିର std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ସ୍ଥିର std::vector<ggml_backend_buffer_type> bufts;
    ସ୍ଥିର std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;
    ଯଦି (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = ନୂତନ ggml_backend_metal_buffer_type{
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
                    /* .alloc_buffer_n      = */ ଖାଲି,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ ଖାଲି,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ &bufts[device];
}

// backend

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ଫେରନ୍ତୁ ggml_metal_get_name(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    ମୁକ୍ତ(backend);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    ଯଦି (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    ଫେରନ୍ତୁ ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ଫେରନ୍ତୁ ggml_metal_graph_compute(ctx, cgraph);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_set_n_cb(ggml_backend_t backend, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

ସ୍ଥିର ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ଖାଲି,
    /* .get_tensor_2d_async     = */ ଖାଲି,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ଖାଲି,
    /* .graph_plan_free         = */ ଖାଲି,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ଖାଲି,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

ସ୍ଥିର ggml_guid_t ggml_backend_metal_guid(ଶୂନ୍ୟ) {
    ସ୍ଥିର ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    ଫେରନ୍ତୁ &guid;
}

ggml_backend_t ggml_backend_metal_init(ଶୂନ୍ୟ) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ଯଦି (ctx == ଖାଲି) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(ଆକାର(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    ଫେରନ୍ତୁ backend;
}

ତାର୍କିକ ggml_backend_is_metal(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != ଖାଲି && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

ଶୂନ୍ୟ ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, ଶୂନ୍ୟ * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

ତାର୍କିକ ggml_backend_metal_supports_family(ggml_backend_t backend, ପୂର୍ଣ୍ଣସଂଖ୍ୟା family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ଫେରନ୍ତୁ ggml_metal_supports_family(ctx, family);
}

ଶୂନ୍ୟ ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ଅଚଳ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ଫେରନ୍ତୁ props_dev->name;
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ଫେରନ୍ତୁ ggml_metal_device_get_props(ctx_dev)->desc;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, ମୁକ୍ତ, total);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ ସତ୍ୟ,
        /* .host_buffer          = */ ମିଥ୍ୟା,
        /* .buffer_from_host_ptr = */ ସତ୍ୟ,
        /* .events               = */ ସତ୍ୟ,
        /* .mmap_support         = */ ସତ୍ୟ,
    };
}

ସ୍ଥିର ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ଯଦି (ctx == ଖାଲି) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(ଆକାର(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    ଫେରନ୍ତୁ backend;

    GGML_UNUSED(params);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ଅଚଳ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ଫେରନ୍ତୁ props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, ଶୂନ୍ୟ * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    ଅଚଳ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ଫେରନ୍ତୁ ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ଫେରନ୍ତୁ ggml_metal_device_supports_op(ctx_dev, op);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

ସ୍ଥିର int64_t get_op_batch_size(ଅଚଳ ggml_tensor * op) {
    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            ଫେରନ୍ତୁ op->ne[1];
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            ଫେରନ୍ତୁ op->ne[2];
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ggml_nrows(op);
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ଫେରନ୍ତୁ (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

ସ୍ଥିର ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = ନୂତନ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    ଫେରନ୍ତୁ ev;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

ସ୍ଥିର ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ ଖାଲି,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

ସଂରଚନା ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_backend_metal_reg * ggml_backend_metal_reg_t;

ସ୍ଥିର ggml_backend_metal_reg_t ggml_backend_metal_reg_init(ଶୂନ୍ୟ) {
    ggml_backend_metal_reg_t ctx = ନୂତନ ସଂରଚନା ggml_backend_metal_reg;

    ଫେରନ୍ତୁ ctx;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

ସଂରଚନା ggml_backend_metal_reg_deleter {
    ଶୂନ୍ୟ operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା std::unique_ptr<ସଂରଚନା ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

ସ୍ଥିର size_t ggml_backend_metal_reg_device_ଗଣନା(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    ଫେରନ୍ତୁ ctx->devices.size();
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    ଫେରନ୍ତୁ ctx->devices[index];
}

ସ୍ଥିର ggml_backend_feature g_ggml_backend_metal_features[] = {
#ଯଦି defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { ଖାଲି, ଖାଲି },
};

ସ୍ଥିର ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_tuning_set_fa_vec_override(ପୂର୍ଣ୍ଣସଂଖ୍ୟା Q, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_tuning_clear_fa_vec_override(ଶୂନ୍ୟ) {
    ggml_metal_tuning::fa_vec_clear_override();
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    ଫେରନ୍ତୁ ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    ଫେରନ୍ତୁ ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_metal_tuning_fa_vec_baseline_ne(ପୂର୍ଣ୍ଣସଂଖ୍ୟା dk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dv) {
    ଫେରନ୍ତୁ ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ଫେରନ୍ତୁ ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ * ggml_backend_fusion_t;

ସ୍ଥିର ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((ସଂରଚନା ggml_metal_fusion_info *) finfo);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((ସଂରଚନା ggml_metal_fusion_info *) finfo);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ଅଚଳ ଅକ୍ଷର ** labels, uint64_t * counts, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {
    ଫେରନ୍ତୁ ggml_metal_fusion_info_stats_get((ସଂରଚନା ggml_metal_fusion_info *) finfo, labels, counts, n);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, ତାର୍କିକ enabled) {
    ggml_metal_fusion_info_set_enabled((ସଂରଚନା ggml_metal_fusion_info *) finfo, enabled);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର * name) {
    ଯଦି (strcmp(name, "ggml_backend_get_features") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_get_features;
    }
    ଯଦି (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    ଯଦି (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    ଯଦି (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    ଯଦି (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    ଯଦି (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    ଯଦି (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    ଯଦି (strcmp(name, "ggml_backend_fusion_get") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_fusion_get;
    }
    ଯଦି (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_fusion_stats_init;
    }
    ଯଦି (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_fusion_stats_reset;
    }
    ଯଦି (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_fusion_stats_get;
    }
    ଯଦି (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_metal_fusion_set_enabled;
    }

    ଫେରନ୍ତୁ ଖାଲି;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

ସ୍ଥିର ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ଫେରନ୍ତୁ ନୂତନ ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

ସଂରଚନା ggml_backend_device_deleter {
    ଶୂନ୍ୟ operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(ଶୂନ୍ୟ) {
    ସ୍ଥିର ggml_backend_reg reg;
    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;

    {
        ସ୍ଥିର std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ଅଚଳ ଅକ୍ଷର * env = getenv("GGML_METAL_DEVICES");
        ଯଦି (env) {
            g_devices = atoi(env);
        }

        ସ୍ଥିର std::vector<ggml_backend_device_ptr> devs;

        ଯଦି (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", ସତ୍ୟ);

            ସ୍ଥିର ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < g_devices; ++i) {
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
