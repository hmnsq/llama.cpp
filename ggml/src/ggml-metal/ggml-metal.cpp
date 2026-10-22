#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-device.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-fusion.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-tuning.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_NAME "MTL"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
ਸਥਿਰ ਪੂਰਨ g_devices = 1;

// forward declaration
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_metal(ggml_backend_ਬਫਰ_t ਬਫਰ);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_shared_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_free(ctx);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_metal_ਬਫਰ_shared_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(ggml_metal_ਬਫਰ_is_shared(ctx));

    ਵਾਪਸ ggml_metal_ਬਫਰ_get_ਅਧਾਰ(ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_shared_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_memset_tensor(ctx, tensor, value, ਵਿੱਥ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_shared_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_ਸੈੱਟ_tensor(ctx, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_shared_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_get_tensor(ctx, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_ਬਫਰ_shared_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(ggml_metal_ਬਫਰ_is_shared(ctx));

    ਜੇ (!ggml_backend_ਬਫਰ_is_metal(src->ਬਫਰ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ggml_metal_ਬਫਰ_cpy_tensor(ctx, src, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_shared_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_clear(ctx, value);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_metal_ਬਫਰ_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_ਬਫਰ_shared_free_ਬਫਰ,
    /* .get_base      = */ ggml_backend_metal_ਬਫਰ_shared_get_ਅਧਾਰ,
    /* .init_tensor   = */ ਨਿਰਾ,
    /* .memset_tensor = */ ggml_backend_metal_ਬਫਰ_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_ਬਫਰ_shared_ਸੈੱਟ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_ਬਫਰ_shared_get_tensor,
    /* .set_tensor_2d = */ ਨਿਰਾ,
    /* .get_tensor_2d = */ ਨਿਰਾ,
    /* .cpy_tensor    = */ ggml_backend_metal_ਬਫਰ_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_ਬਫਰ_shared_clear,
    /* .reset         = */ ਨਿਰਾ,
};

// private buffer

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(!ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_free(ctx);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(!ggml_metal_ਬਫਰ_is_shared(ctx));

    ਵਾਪਸ ggml_metal_ਬਫਰ_get_ਅਧਾਰ(ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(!ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_memset_tensor(ctx, tensor, value, ਵਿੱਥ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(!ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_ਸੈੱਟ_tensor(ctx, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(!ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_get_tensor(ctx, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(!ggml_metal_ਬਫਰ_is_shared(ctx));

    ਜੇ (!ggml_backend_ਬਫਰ_is_metal(src->ਬਫਰ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ggml_metal_ਬਫਰ_cpy_tensor(ctx, src, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ggml_metal_ਬਫਰ_t ctx = (ggml_metal_ਬਫਰ_t)ਬਫਰ->context;

    GGML_ASSERT(!ggml_metal_ਬਫਰ_is_shared(ctx));

    ggml_metal_ਬਫਰ_clear(ctx, value);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_i = {
    /* .free_buffer   = */ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_free_ਬਫਰ,
    /* .get_base      = */ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_get_ਅਧਾਰ,
    /* .init_tensor   = */ ਨਿਰਾ,
    /* .memset_tensor = */ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_ਸੈੱਟ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_get_tensor,
    /* .set_tensor_2d = */ ਨਿਰਾ,
    /* .get_tensor_2d = */ ਨਿਰਾ,
    /* .cpy_tensor    = */ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_clear,
    /* .reset         = */ ਨਿਰਾ,
};

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_metal(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ਬਫਰ->iface.free_ਬਫਰ == ggml_backend_metal_ਬਫਰ_shared_free_ਬਫਰ ||
           ਬਫਰ->iface.free_ਬਫਰ == ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_free_ਬਫਰ;
}

//
// buffer types
//

ਬਣਤਰ ggml_backend_metal_ਬਫਰ_type {
    ਪੂਰਨ device;
    std::string name;
};

ਬਣਤਰ ggml_backend_metal_ਬਫਰ_type_deleter {
    ਖਾਲੀ ਸੰਚਾਲਕ()(ggml_backend_metal_ਬਫਰ_type * ctx) ਅਚਲ {
        ਮਿਟਾਓ ctx;
    }
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::unique_ptr<ggml_backend_metal_ਬਫਰ_type, ggml_backend_metal_ਬਫਰ_type_deleter> ggml_backend_metal_ਬਫਰ_type_ptr;

// common method for allocating shread or private Metal buffers
ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_metal_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size, ਤਾਰਕਿਕ shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_ਬਫਰ_t res = ggml_metal_ਬਫਰ_init(ctx_dev, size, shared);

    ਜੇ (res == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ggml_backend_ਬਫਰ_i buf_i = ggml_metal_ਬਫਰ_is_shared(res)
        ? ggml_backend_metal_ਬਫਰ_shared_i
        : ggml_backend_metal_ਬਫਰ_ਨਿੱਜੀ_i;

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, buf_i, res, size);
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    ਚੋਣ (tensor->op) {
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUMSUM:
        ਹਾਲਤ GGML_OP_ARGSORT:
            {
                res *= 2;
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            {
                res = 2*ਆਕਾਰ(int32_t)*ggml_nelements(tensor->src[0]);
            } ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
    }

    ਵਾਪਸ res;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

// default (shared) buffer type

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_ਬਫਰ_type_shared_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_metal_ਬਫਰ_type * ctx = (ggml_backend_metal_ਬਫਰ_type *)buft->context;

    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_metal_ਬਫਰ_type_shared_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ਵਾਪਸ ggml_backend_metal_ਬਫਰ_type_alloc_ਬਫਰ(buft, size, ਸੱਚ);
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_shared_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ 32;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_shared_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ਵਾਪਸ ggml_metal_device_get_props(ctx_dev)->max_ਬਫਰ_ਮਾਪ;
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_shared_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_backend_metal_ਬਫਰ_type_get_alloc_ਮਾਪ(buft, tensor);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_ਬਫਰ_type_shared_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_metal_ਬਫਰ_type_shared(ਪੂਰਨ device) {
    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

    ਸਥਿਰ std::vector<ggml_backend_ਬਫਰ_type> bufts;
    ਸਥਿਰ std::vector<ggml_backend_metal_ਬਫਰ_type_ptr> ctxs;

    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;
    ਜੇ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ਲਈ (ਪੂਰਨ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_ਬਫਰ_type * raw_ctx =
                ਨਵਾਂ ggml_backend_metal_ਬਫਰ_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_ਬਫਰ_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_ਬਫਰ_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_ਬਫਰ_type_shared_alloc_ਬਫਰ,
                    /* .alloc_buffer_n      = */ ਨਿਰਾ,
                    /* .get_alignment       = */ ggml_backend_metal_ਬਫਰ_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_ਬਫਰ_type_shared_get_max_ਮਾਪ,
                    /* .get_alloc_size      = */ ggml_backend_metal_ਬਫਰ_type_shared_get_alloc_ਮਾਪ,
                    /* .get_alloc_size_n    = */ ਨਿਰਾ,
                    /* .is_host             = */ ggml_backend_metal_ਬਫਰ_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ &bufts[device];
}

// default (private) buffer type

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_metal_ਬਫਰ_type * ctx = (ggml_backend_metal_ਬਫਰ_type *)buft->context;

    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ਵਾਪਸ ggml_backend_metal_ਬਫਰ_type_alloc_ਬਫਰ(buft, size, ਝੂਠ);
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ 32;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ਵਾਪਸ ggml_metal_device_get_props(ctx_dev)->max_ਬਫਰ_ਮਾਪ;
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_backend_metal_ਬਫਰ_type_get_alloc_ਮਾਪ(buft, tensor);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ(ਪੂਰਨ device) {
    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

    ਸਥਿਰ std::vector<ggml_backend_ਬਫਰ_type> bufts;
    ਸਥਿਰ std::vector<ggml_backend_metal_ਬਫਰ_type_ptr> ctxs;

    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;
    ਜੇ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ਲਈ (ਪੂਰਨ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_ਬਫਰ_type * raw_ctx = ਨਵਾਂ ggml_backend_metal_ਬਫਰ_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_ਬਫਰ_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_alloc_ਬਫਰ,
                    /* .alloc_buffer_n      = */ ਨਿਰਾ,
                    /* .get_alignment       = */ ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_max_ਮਾਪ,
                    /* .get_alloc_size      = */ ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_alloc_ਮਾਪ,
                    /* .get_alloc_size_n    = */ ਨਿਰਾ,
                    /* .is_host             = */ ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ &bufts[device];
}

// mapped buffer type

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_ਬਫਰ_type_mapped_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_metal_ਬਫਰ_type * ctx = (ggml_backend_metal_ਬਫਰ_type *)buft->context;

    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_metal_ਬਫਰ_type_mapped_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    ਵਾਪਸ ggml_backend_metal_ਬਫਰ_type_alloc_ਬਫਰ(buft, size, ਸੱਚ);
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_mapped_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ 32;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_mapped_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ਵਾਪਸ ggml_metal_device_get_props(ctx_dev)->max_ਬਫਰ_ਮਾਪ;
}

ਸਥਿਰ size_t ggml_backend_metal_ਬਫਰ_type_mapped_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_backend_metal_ਬਫਰ_type_get_alloc_ਮਾਪ(buft, tensor);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_ਬਫਰ_type_mapped_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_metal_ਬਫਰ_type_mapped(ਪੂਰਨ device) {
    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

    ਸਥਿਰ std::vector<ggml_backend_ਬਫਰ_type> bufts;
    ਸਥਿਰ std::vector<ggml_backend_metal_ਬਫਰ_type_ptr> ctxs;

    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;
    ਜੇ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ਲਈ (ਪੂਰਨ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_ਬਫਰ_type * raw_ctx = ਨਵਾਂ ggml_backend_metal_ਬਫਰ_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_ਬਫਰ_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_ਬਫਰ_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_ਬਫਰ_type_mapped_alloc_ਬਫਰ,
                    /* .alloc_buffer_n      = */ ਨਿਰਾ,
                    /* .get_alignment       = */ ggml_backend_metal_ਬਫਰ_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_ਬਫਰ_type_mapped_get_max_ਮਾਪ,
                    /* .get_alloc_size      = */ ggml_backend_metal_ਬਫਰ_type_mapped_get_alloc_ਮਾਪ,
                    /* .get_alloc_size_n    = */ ਨਿਰਾ,
                    /* .is_host             = */ ggml_backend_metal_ਬਫਰ_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ &bufts[device];
}

// backend

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ਵਾਪਸ ggml_metal_get_name(ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    ਮੁਕਤ(backend);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਸੈੱਟ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ਸੈੱਟ_tensor_async(ctx, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਜੇ (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_backend_ਬਫਰ_is_metal(src->ਬਫਰ) || !ggml_backend_ਬਫਰ_is_metal(dst->ਬਫਰ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    ਵਾਪਸ ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_metal_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ਵਾਪਸ ggml_metal_ਗ੍ਰਾਫ_compute(ctx, cgraph);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_event_ਰਿਕਾਰਡ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_ਰਿਕਾਰਡ(ctx, ev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਗ੍ਰਾਫ_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_ਗ੍ਰਾਫ_optimize_params * params) {
    GGML_ASSERT(params && params->ਜੋੜੋ_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_ਜੋੜੋ_alloc_deps(params->user_ਅੰਕੜੇ, params->ਜੋੜੋ_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ਗ੍ਰਾਫ_optimize(ctx, cgraph);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_ਸੈੱਟ_n_cb(ggml_backend_t backend, ਪੂਰਨ n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ਸੈੱਟ_n_cb(ctx, n_cb);
}

ਸਥਿਰ ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_metal_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ggml_backend_metal_event_ਰਿਕਾਰਡ,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_ਗ੍ਰਾਫ_optimize,
};

ਸਥਿਰ ggml_guid_t ggml_backend_metal_guid(ਖਾਲੀ) {
    ਸਥਿਰ ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    ਵਾਪਸ &guid;
}

ggml_backend_t ggml_backend_metal_init(ਖਾਲੀ) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ਜੇ (ctx == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ggml_backend_t backend = (ggml_backend_t) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_ਸੈੱਟ_n_cb(backend, 1);

    ਵਾਪਸ backend;
}

ਤਾਰਕਿਕ ggml_backend_is_metal(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

ਖਾਲੀ ggml_backend_metal_ਸੈੱਟ_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, ਖਾਲੀ * user_ਅੰਕੜੇ) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ਸੈੱਟ_abort_callback(ctx, abort_callback, user_ਅੰਕੜੇ);
}

ਤਾਰਕਿਕ ggml_backend_metal_supports_family(ggml_backend_t backend, ਪੂਰਨ family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ਵਾਪਸ ggml_metal_supports_family(ctx, family);
}

ਖਾਲੀ ggml_backend_metal_capture_ਅਗਲਾ_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_ਅਗਲਾ_compute(ctx);
}

// backend device

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ਅਚਲ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ਵਾਪਸ props_dev->name;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ਵਾਪਸ ggml_metal_device_get_props(ctx_dev)->desc;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_ਸਮ੍ਰਿਤੀ(ctx_dev, ਮੁਕਤ, ਕੁੱਲ);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);

    props->caps = {
        /* .async                = */ ਸੱਚ,
        /* .host_buffer          = */ ਝੂਠ,
        /* .buffer_from_host_ptr = */ ਸੱਚ,
        /* .events               = */ ਸੱਚ,
        /* .mmap_support         = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ਜੇ (ctx == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ggml_backend_t backend = (ggml_backend_t) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_ਸੈੱਟ_n_cb(backend, 1);

    ਵਾਪਸ backend;

    GGML_ਅਣਵਰਤਿਆ(params);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_metal_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ਅਚਲ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ਵਾਪਸ props_dev->use_shared_buffers ? ggml_backend_metal_ਬਫਰ_type_shared(props_dev->device) : ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ(props_dev->device);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_metal_device_ਬਫਰ_mapped(ggml_backend_dev_t dev, ਖਾਲੀ * ptr, size_t size, size_t max_tensor_ਮਾਪ) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_ਬਫਰ_t res = ggml_metal_ਬਫਰ_map(ctx_dev, ptr, size, max_tensor_ਮਾਪ);

    ਅਚਲ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(ggml_backend_metal_ਬਫਰ_type_mapped(props_dev->device), ggml_backend_metal_ਬਫਰ_shared_i, res, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ਵਾਪਸ ggml_metal_device_supports_op(ctx_dev, op);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_ਬਫਰ_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_ਬਫਰ_type_ਨਿੱਜੀ_get_name ||
        buft->iface.get_name == ggml_backend_metal_ਬਫਰ_type_mapped_get_name);

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ int64_t get_op_batch_ਮਾਪ(ਅਚਲ ggml_tensor * op) {
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ਵਾਪਸ op->ne[1];
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            ਵਾਪਸ op->ne[2];
        ਮੂਲ:
            ਵਾਪਸ ggml_nrows(op);
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ਵਾਪਸ (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_ਮਾਪ(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_ਮਾਪ;
}

ਸਥਿਰ ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = ਨਵਾਂ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    ਵਾਪਸ ev;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    ਮਿਟਾਓ event;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

ਸਥਿਰ ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_ਬਫਰ_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

ਬਣਤਰ ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_backend_metal_reg * ggml_backend_metal_reg_t;

ਸਥਿਰ ggml_backend_metal_reg_t ggml_backend_metal_reg_init(ਖਾਲੀ) {
    ggml_backend_metal_reg_t ctx = ਨਵਾਂ ਬਣਤਰ ggml_backend_metal_reg;

    ਵਾਪਸ ctx;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    ਮਿਟਾਓ ctx;
}

ਬਣਤਰ ggml_backend_metal_reg_deleter {
    ਖਾਲੀ ਸੰਚਾਲਕ()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::unique_ptr<ਬਣਤਰ ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    ਵਾਪਸ GGML_METAL_NAME;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ size_t ggml_backend_metal_reg_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    ਵਾਪਸ ctx->devices.size();
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(ਸੂਚਕਾਂਕ < ctx->devices.size());
    ਵਾਪਸ ctx->devices[ਸੂਚਕਾਂਕ];
}

ਸਥਿਰ ggml_backend_feature g_ggml_backend_metal_features[] = {
#ਜੇ defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#ਜੇ_ਅੰਤ
    { ਨਿਰਾ, ਨਿਰਾ },
};

ਸਥਿਰ ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    ਵਾਪਸ g_ggml_backend_metal_features;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_tuning_ਸੈੱਟ_fa_vec_override(ਪੂਰਨ Q, ਪੂਰਨ NE) {
    ggml_metal_tuning::fa_vec_ਸੈੱਟ_override({ (int8_t) Q, (int8_t) NE });
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_tuning_clear_fa_vec_override(ਖਾਲੀ) {
    ggml_metal_tuning::fa_vec_clear_override();
}

ਸਥਿਰ ਪੂਰਨ ggml_backend_metal_tuning_fa_vec_ne11_ਬਾਲਟੀ(int64_t ne11) {
    ਵਾਪਸ ggml_metal_tuning::fa_vec_ne11_ਬਾਲਟੀ(ne11);
}

ਸਥਿਰ ਪੂਰਨ ggml_backend_metal_tuning_fa_vec_ne01_ਬਾਲਟੀ(int64_t ne01) {
    ਵਾਪਸ ggml_metal_tuning::fa_vec_ne01_ਬਾਲਟੀ(ne01);
}

ਸਥਿਰ ਪੂਰਨ ggml_backend_metal_tuning_fa_vec_baseline_ne(ਪੂਰਨ dk, ਪੂਰਨ dv) {
    ਵਾਪਸ ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ਵਾਪਸ ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ * ggml_backend_fusion_t;

ਸਥਿਰ ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    ਵਾਪਸ ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((ਬਣਤਰ ggml_metal_fusion_info *) finfo);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((ਬਣਤਰ ggml_metal_fusion_info *) finfo);
}

ਸਥਿਰ ਪੂਰਨ ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ਅਚਲ ਅੱਖਰ ** labels, uint64_t * counts, ਪੂਰਨ n) {
    ਵਾਪਸ ggml_metal_fusion_info_stats_get((ਬਣਤਰ ggml_metal_fusion_info *) finfo, labels, counts, n);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_fusion_ਸੈੱਟ_enabled(ggml_backend_fusion_t finfo, ਤਾਰਕਿਕ enabled) {
    ggml_metal_fusion_info_ਸੈੱਟ_enabled((ਬਣਤਰ ggml_metal_fusion_info *) finfo, enabled);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_metal_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (strcmp(name, "ggml_backend_get_features") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_get_features;
    }
    ਜੇ (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_tuning_ਸੈੱਟ_fa_vec_override;
    }
    ਜੇ (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    ਜੇ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_tuning_fa_vec_ne11_ਬਾਲਟੀ;
    }
    ਜੇ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_tuning_fa_vec_ne01_ਬਾਲਟੀ;
    }
    ਜੇ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    ਜੇ (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    ਜੇ (strcmp(name, "ggml_backend_fusion_get") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_fusion_get;
    }
    ਜੇ (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_fusion_stats_init;
    }
    ਜੇ (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_fusion_stats_reset;
    }
    ਜੇ (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_fusion_stats_get;
    }
    ਜੇ (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_metal_fusion_ਸੈੱਟ_enabled;
    }

    ਵਾਪਸ ਨਿਰਾ;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_ਪਤਾ,
};

ਸਥਿਰ ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, ਪੂਰਨ device) {
    ਵਾਪਸ ਨਵਾਂ ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    ਮਿਟਾਓ dev;
}

ਬਣਤਰ ggml_backend_device_deleter {
    ਖਾਲੀ ਸੰਚਾਲਕ()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(ਖਾਲੀ) {
    ਸਥਿਰ ggml_backend_reg reg;
    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;

    {
        ਸਥਿਰ std::mutex mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

        ਅਚਲ ਅੱਖਰ * env = getenv("GGML_METAL_DEVICES");
        ਜੇ (env) {
            g_devices = atoi(env);
        }

        ਸਥਿਰ std::vector<ggml_backend_device_ptr> devs;

        ਜੇ (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", ਸੱਚ);

            ਸਥਿਰ ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            ਲਈ (ਪੂਰਨ i = 0; i < g_devices; ++i) {
                ਸਵੈਚਾਲਿਤ * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
