#ಅನ್ತರ್ಭಾವ "ggml-metal.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-fusion.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-context.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-ops.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-tuning.h"

#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ g_devices = 1;

// forward declaration
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ಪ್ರತ್ಯಾಗಮ ggml_metal_buffer_get_base(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ಯದಿ (!ggml_backend_buffer_is_metal(src->buffer)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

ಸ್ಥಿರ ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ ರಿಕ್ತ,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ ರಿಕ್ತ,
    /* .get_tensor_2d = */ ರಿಕ್ತ,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ ರಿಕ್ತ,
};

// private buffer

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ಪ್ರತ್ಯಾಗಮ ggml_metal_buffer_get_base(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ಯದಿ (!ggml_backend_buffer_is_metal(src->buffer)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

ಸ್ಥಿರ ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ ರಿಕ್ತ,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ ರಿಕ್ತ,
    /* .get_tensor_2d = */ ರಿಕ್ತ,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

ಸಂರಚನಾ ggml_backend_metal_buffer_type {
    ಪೂರ್ಣಾಙ್ಕ device;
    std::string name;
};

ಸಂರಚನಾ ggml_backend_metal_buffer_type_deleter {
    ಶೂನ್ಯ operator()(ggml_backend_metal_buffer_type * ctx) ಧ್ರುವ {
        delete ctx;
    }
};

ಪ್ರಕಾರನಿರ್ದೇಶ std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, ತಾರ್ಕಿಕ shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    ಯದಿ (res == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, buf_i, res, size);
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    ವಿಕಲ್ಪ (tensor->op) {
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CUMSUM:
        ಪಕ್ಷ GGML_OP_ARGSORT:
            {
                res *= 2;
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ತುದಿ_K:
            {
                res = 2*ಪರಿಮಾಣ(int32_t)*ggml_nelements(tensor->src[0]);
            } ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }

    ಪ್ರತ್ಯಾಗಮ res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    ಪ್ರತ್ಯಾಗಮ ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, ಸತ್ಯ);
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ 32;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(ಪೂರ್ಣಾಙ್ಕ device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ಸ್ಥಿರ std::vector<ggml_backend_buffer_type> bufts;
    ಸ್ಥಿರ std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;
    ಯದಿ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                ನೂತನ ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ ರಿಕ್ತ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ &bufts[device];
}

// default (private) buffer type

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    ಪ್ರತ್ಯಾಗಮ ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, ಅಸತ್ಯ);
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ 32;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(ಪೂರ್ಣಾಙ್ಕ device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ಸ್ಥಿರ std::vector<ggml_backend_buffer_type> bufts;
    ಸ್ಥಿರ std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;
    ಯದಿ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = ನೂತನ ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ ರಿಕ್ತ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ &bufts[device];
}

// mapped buffer type

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    ಪ್ರತ್ಯಾಗಮ ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    ಪ್ರತ್ಯಾಗಮ ggml_backend_metal_buffer_type_alloc_buffer(buft, size, ಸತ್ಯ);
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ 32;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

ಸ್ಥಿರ size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(ಪೂರ್ಣಾಙ್ಕ device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ಸ್ಥಿರ std::vector<ggml_backend_buffer_type> bufts;
    ಸ್ಥಿರ std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;
    ಯದಿ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = ನೂತನ ggml_backend_metal_buffer_type{
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
                    /* .alloc_buffer_n      = */ ರಿಕ್ತ,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ &bufts[device];
}

// backend

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_get_name(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    ಮುಕ್ತ(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ಯದಿ (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_graph_compute(ctx, cgraph);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_set_n_cb(ggml_backend_t backend, ಪೂರ್ಣಾಙ್ಕ n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .get_tensor_2d_async     = */ ರಿಕ್ತ,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ರಿಕ್ತ,
    /* .graph_plan_free         = */ ರಿಕ್ತ,
    /* .graph_plan_update       = */ ರಿಕ್ತ,
    /* .graph_plan_compute      = */ ರಿಕ್ತ,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_metal_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    ಪ್ರತ್ಯಾಗಮ &guid;
}

ggml_backend_t ggml_backend_metal_init(ಶೂನ್ಯ) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ಯದಿ (ctx == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(ಪರಿಮಾಣ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    ಪ್ರತ್ಯಾಗಮ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_metal(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend != ರಿಕ್ತ && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

ಶೂನ್ಯ ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, ಶೂನ್ಯ * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

ತಾರ್ಕಿಕ ggml_backend_metal_supports_family(ggml_backend_t backend, ಪೂರ್ಣಾಙ್ಕ family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_supports_family(ctx, family);
}

ಶೂನ್ಯ ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಧ್ರುವ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ಪ್ರತ್ಯಾಗಮ props_dev->name;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_device_get_props(ctx_dev)->desc;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, ಮುಕ್ತ, total);
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ ಸತ್ಯ,
        /* .host_buffer          = */ ಅಸತ್ಯ,
        /* .buffer_from_host_ptr = */ ಸತ್ಯ,
        /* .events               = */ ಸತ್ಯ,
        /* .mmap_support         = */ ಸತ್ಯ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ಯದಿ (ctx == ರಿಕ್ತ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(ಪರಿಮಾಣ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    ಪ್ರತ್ಯಾಗಮ backend;

    GGML_UNUSED(params);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಧ್ರುವ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ಪ್ರತ್ಯಾಗಮ props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    ಧ್ರುವ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_device_supports_op(ctx_dev, op);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ int64_t get_op_batch_size(ಧ್ರುವ ggml_tensor * op) {
    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            ಪ್ರತ್ಯಾಗಮ op->ne[1];
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            ಪ್ರತ್ಯಾಗಮ op->ne[2];
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ggml_nrows(op);
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಪ್ರತ್ಯಾಗಮ (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = ನೂತನ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    ಪ್ರತ್ಯಾಗಮ ev;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

ಸ್ಥಿರ ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ ರಿಕ್ತ,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

ಸಂರಚನಾ ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ ggml_backend_metal_reg * ggml_backend_metal_reg_t;

ಸ್ಥಿರ ggml_backend_metal_reg_t ggml_backend_metal_reg_init(ಶೂನ್ಯ) {
    ggml_backend_metal_reg_t ctx = ನೂತನ ಸಂರಚನಾ ggml_backend_metal_reg;

    ಪ್ರತ್ಯಾಗಮ ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

ಸಂರಚನಾ ggml_backend_metal_reg_deleter {
    ಶೂನ್ಯ operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

ಪ್ರಕಾರನಿರ್ದೇಶ std::unique_ptr<ಸಂರಚನಾ ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ size_t ggml_backend_metal_reg_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    ಪ್ರತ್ಯಾಗಮ ctx->devices.size();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    ಪ್ರತ್ಯಾಗಮ ctx->devices[index];
}

ಸ್ಥಿರ ggml_backend_feature g_ggml_backend_metal_features[] = {
#ಯದಿ defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { ರಿಕ್ತ, ರಿಕ್ತ },
};

ಸ್ಥಿರ ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_tuning_set_fa_vec_override(ಪೂರ್ಣಾಙ್ಕ Q, ಪೂರ್ಣಾಙ್ಕ NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_tuning_clear_fa_vec_override(ಶೂನ್ಯ) {
    ggml_metal_tuning::fa_vec_clear_override();
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    ಪ್ರತ್ಯಾಗಮ ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    ಪ್ರತ್ಯಾಗಮ ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_metal_tuning_fa_vec_baseline_ne(ಪೂರ್ಣಾಙ್ಕ dk, ಪೂರ್ಣಾಙ್ಕ dv) {
    ಪ್ರತ್ಯಾಗಮ ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ * ggml_backend_fusion_t;

ಸ್ಥಿರ ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((ಸಂರಚನಾ ggml_metal_fusion_info *) finfo);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((ಸಂರಚನಾ ggml_metal_fusion_info *) finfo);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ಧ್ರುವ ವರ್ಣ ** labels, uint64_t * counts, ಪೂರ್ಣಾಙ್ಕ n) {
    ಪ್ರತ್ಯಾಗಮ ggml_metal_fusion_info_stats_get((ಸಂರಚನಾ ggml_metal_fusion_info *) finfo, labels, counts, n);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, ತಾರ್ಕಿಕ enabled) {
    ggml_metal_fusion_info_set_enabled((ಸಂರಚನಾ ggml_metal_fusion_info *) finfo, enabled);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, ಧ್ರುವ ವರ್ಣ * name) {
    ಯದಿ (strcmp(name, "ggml_backend_get_features") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_get_features;
    }
    ಯದಿ (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    ಯದಿ (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    ಯದಿ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    ಯದಿ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    ಯದಿ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    ಯದಿ (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    ಯದಿ (strcmp(name, "ggml_backend_fusion_get") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_get;
    }
    ಯದಿ (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_stats_init;
    }
    ಯದಿ (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_stats_reset;
    }
    ಯದಿ (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_stats_get;
    }
    ಯದಿ (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_set_enabled;
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, ಪೂರ್ಣಾಙ್ಕ device) {
    ಪ್ರತ್ಯಾಗಮ ನೂತನ ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

ಸಂರಚನಾ ggml_backend_device_deleter {
    ಶೂನ್ಯ operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

ಪ್ರಕಾರನಿರ್ದೇಶ std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_reg reg;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;

    {
        ಸ್ಥಿರ std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ಧ್ರುವ ವರ್ಣ * env = getenv("GGML_METAL_DEVICES");
        ಯದಿ (env) {
            g_devices = atoi(env);
        }

        ಸ್ಥಿರ std::vector<ggml_backend_device_ptr> devs;

        ಯದಿ (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", ಸತ್ಯ);

            ಸ್ಥಿರ ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < g_devices; ++i) {
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

        initialized = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
