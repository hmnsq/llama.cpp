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

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_NAME "MTL"
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ g_devices = 1;

// forward declaration
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_metal(ggml_backend_ಬಫರ್_t ಬಫರ್);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_shared_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_free(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_metal_ಬಫರ್_shared_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(ggml_metal_ಬಫರ್_is_shared(ctx));

    ಹಿಂತಿರುಗಿ ggml_metal_ಬಫರ್_get_ತಳ(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_shared_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_memset_tensor(ctx, tensor, value, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_shared_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_ಗಣ_tensor(ctx, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_shared_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_get_tensor(ctx, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_ಬಫರ್_shared_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(ggml_metal_ಬಫರ್_is_shared(ctx));

    ಆದರೆ (!ggml_backend_ಬಫರ್_is_metal(src->ಬಫರ್)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ggml_metal_ಬಫರ್_cpy_tensor(ctx, src, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_shared_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_clear(ctx, value);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_metal_ಬಫರ್_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_ಬಫರ್_shared_free_ಬಫರ್,
    /* .get_base      = */ ggml_backend_metal_ಬಫರ್_shared_get_ತಳ,
    /* .init_tensor   = */ ಖಾಲಿ,
    /* .memset_tensor = */ ggml_backend_metal_ಬಫರ್_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_ಬಫರ್_shared_ಗಣ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_ಬಫರ್_shared_get_tensor,
    /* .set_tensor_2d = */ ಖಾಲಿ,
    /* .get_tensor_2d = */ ಖಾಲಿ,
    /* .cpy_tensor    = */ ggml_backend_metal_ಬಫರ್_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_ಬಫರ್_shared_clear,
    /* .reset         = */ ಖಾಲಿ,
};

// private buffer

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(!ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_free(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(!ggml_metal_ಬಫರ್_is_shared(ctx));

    ಹಿಂತಿರುಗಿ ggml_metal_ಬಫರ್_get_ತಳ(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(!ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_memset_tensor(ctx, tensor, value, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(!ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_ಗಣ_tensor(ctx, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(!ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_get_tensor(ctx, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(!ggml_metal_ಬಫರ್_is_shared(ctx));

    ಆದರೆ (!ggml_backend_ಬಫರ್_is_metal(src->ಬಫರ್)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ggml_metal_ಬಫರ್_cpy_tensor(ctx, src, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ggml_metal_ಬಫರ್_t ctx = (ggml_metal_ಬಫರ್_t)ಬಫರ್->context;

    GGML_ASSERT(!ggml_metal_ಬಫರ್_is_shared(ctx));

    ggml_metal_ಬಫರ್_clear(ctx, value);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_i = {
    /* .free_buffer   = */ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_free_ಬಫರ್,
    /* .get_base      = */ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_get_ತಳ,
    /* .init_tensor   = */ ಖಾಲಿ,
    /* .memset_tensor = */ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_ಗಣ_tensor,
    /* .get_tensor    = */ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_get_tensor,
    /* .set_tensor_2d = */ ಖಾಲಿ,
    /* .get_tensor_2d = */ ಖಾಲಿ,
    /* .cpy_tensor    = */ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_clear,
    /* .reset         = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_metal(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ಬಫರ್->iface.free_ಬಫರ್ == ggml_backend_metal_ಬಫರ್_shared_free_ಬಫರ್ ||
           ಬಫರ್->iface.free_ಬಫರ್ == ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_free_ಬಫರ್;
}

//
// buffer types
//

ರಚನೆ ggml_backend_metal_ಬಫರ್_type {
    ಪೂರ್ಣಾಂಕ device;
    std::string name;
};

ರಚನೆ ggml_backend_metal_ಬಫರ್_type_deleter {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ggml_backend_metal_ಬಫರ್_type * ctx) ಸ್ಥಿರಾಂಕ {
        ಅಳಿಸು ctx;
    }
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::unique_ptr<ggml_backend_metal_ಬಫರ್_type, ggml_backend_metal_ಬಫರ್_type_deleter> ggml_backend_metal_ಬಫರ್_type_ptr;

// common method for allocating shread or private Metal buffers
ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_metal_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size, ತಾರ್ಕಿಕ shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_ಬಫರ್_t res = ggml_metal_ಬಫರ್_init(ctx_dev, size, shared);

    ಆದರೆ (res == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ggml_backend_ಬಫರ್_i buf_i = ggml_metal_ಬಫರ್_is_shared(res)
        ? ggml_backend_metal_ಬಫರ್_shared_i
        : ggml_backend_metal_ಬಫರ್_ಖಾಸಗಿ_i;

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, buf_i, res, size);
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    ಆಯ್ಕೆ (tensor->op) {
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_amax(tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            {
                res *= 2;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            {
                res = 2*ಗಾತ್ರ(int32_t)*ggml_nelements(tensor->src[0]);
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
    }

    ಹಿಂತಿರುಗಿ res;

    GGML_ಬಳಸದ(buft);
}

// default (shared) buffer type

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_ಬಫರ್_type_shared_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_metal_ಬಫರ್_type * ctx = (ggml_backend_metal_ಬಫರ್_type *)buft->context;

    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_metal_ಬಫರ್_type_shared_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ಹಿಂತಿರುಗಿ ggml_backend_metal_ಬಫರ್_type_alloc_ಬಫರ್(buft, size, ನಿಜ);
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_shared_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ 32;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_shared_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ಹಿಂತಿರುಗಿ ggml_metal_device_get_props(ctx_dev)->max_ಬಫರ್_ಗಾತ್ರ;
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_shared_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_backend_metal_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(buft, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_ಬಫರ್_type_shared_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_metal_ಬಫರ್_type_shared(ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);

    ಸ್ಥಿರ std::vector<ggml_backend_ಬಫರ್_type> bufts;
    ಸ್ಥಿರ std::vector<ggml_backend_metal_ಬಫರ್_type_ptr> ctxs;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;
    ಆದರೆ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_ಬಫರ್_type * raw_ctx =
                ಹೊಸ ggml_backend_metal_ಬಫರ್_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_ಬಫರ್_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_ಬಫರ್_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_ಬಫರ್_type_shared_alloc_ಬಫರ್,
                    /* .alloc_buffer_n      = */ ಖಾಲಿ,
                    /* .get_alignment       = */ ggml_backend_metal_ಬಫರ್_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_ಬಫರ್_type_shared_get_max_ಗಾತ್ರ,
                    /* .get_alloc_size      = */ ggml_backend_metal_ಬಫರ್_type_shared_get_alloc_ಗಾತ್ರ,
                    /* .get_alloc_size_n    = */ ಖಾಲಿ,
                    /* .is_host             = */ ggml_backend_metal_ಬಫರ್_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &bufts[device];
}

// default (private) buffer type

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_metal_ಬಫರ್_type * ctx = (ggml_backend_metal_ಬಫರ್_type *)buft->context;

    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ಹಿಂತಿರುಗಿ ggml_backend_metal_ಬಫರ್_type_alloc_ಬಫರ್(buft, size, ಸುಳ್ಳು);
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ 32;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ಹಿಂತಿರುಗಿ ggml_metal_device_get_props(ctx_dev)->max_ಬಫರ್_ಗಾತ್ರ;
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_backend_metal_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(buft, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ(ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);

    ಸ್ಥಿರ std::vector<ggml_backend_ಬಫರ್_type> bufts;
    ಸ್ಥಿರ std::vector<ggml_backend_metal_ಬಫರ್_type_ptr> ctxs;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;
    ಆದರೆ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_ಬಫರ್_type * raw_ctx = ಹೊಸ ggml_backend_metal_ಬಫರ್_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_ಬಫರ್_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_alloc_ಬಫರ್,
                    /* .alloc_buffer_n      = */ ಖಾಲಿ,
                    /* .get_alignment       = */ ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_max_ಗಾತ್ರ,
                    /* .get_alloc_size      = */ ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_alloc_ಗಾತ್ರ,
                    /* .get_alloc_size_n    = */ ಖಾಲಿ,
                    /* .is_host             = */ ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &bufts[device];
}

// mapped buffer type

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_ಬಫರ್_type_mapped_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_metal_ಬಫರ್_type * ctx = (ggml_backend_metal_ಬಫರ್_type *)buft->context;

    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_metal_ಬಫರ್_type_mapped_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    ಹಿಂತಿರುಗಿ ggml_backend_metal_ಬಫರ್_type_alloc_ಬಫರ್(buft, size, ನಿಜ);
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_mapped_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ 32;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_mapped_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    ಹಿಂತಿರುಗಿ ggml_metal_device_get_props(ctx_dev)->max_ಬಫರ್_ಗಾತ್ರ;
}

ಸ್ಥಿರ size_t ggml_backend_metal_ಬಫರ್_type_mapped_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_backend_metal_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(buft, tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_ಬಫರ್_type_mapped_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_metal_ಬಫರ್_type_mapped(ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);

    ಸ್ಥಿರ std::vector<ggml_backend_ಬಫರ್_type> bufts;
    ಸ್ಥಿರ std::vector<ggml_backend_metal_ಬಫರ್_type_ptr> ctxs;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;
    ಆದರೆ (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < g_devices; ++i) {
            ggml_backend_metal_ಬಫರ್_type * raw_ctx = ಹೊಸ ggml_backend_metal_ಬಫರ್_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_ಬಫರ್_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_ಬಫರ್_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_ಬಫರ್_type_mapped_alloc_ಬಫರ್,
                    /* .alloc_buffer_n      = */ ಖಾಲಿ,
                    /* .get_alignment       = */ ggml_backend_metal_ಬಫರ್_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_ಬಫರ್_type_mapped_get_max_ಗಾತ್ರ,
                    /* .get_alloc_size      = */ ggml_backend_metal_ಬಫರ್_type_mapped_get_alloc_ಗಾತ್ರ,
                    /* .get_alloc_size_n    = */ ಖಾಲಿ,
                    /* .is_host             = */ ggml_backend_metal_ಬಫರ್_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &bufts[device];
}

// backend

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ಹಿಂತಿರುಗಿ ggml_metal_get_name(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    ಬಿಡುಗಡೆ(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಗಣ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ಗಣ_tensor_async(ctx, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_get_tensor_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಆದರೆ (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_backend_ಬಫರ್_is_metal(src->ಬಫರ್) || !ggml_backend_ಬಫರ್_is_metal(dst->ಬಫರ್)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    ಹಿಂತಿರುಗಿ ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_metal_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ಹಿಂತಿರುಗಿ ggml_metal_ನಕ್ಷೆ_compute(ctx, cgraph);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_event_ದಾಖಲೆ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_ದಾಖಲೆ(ctx, ev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ನಕ್ಷೆ_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_ನಕ್ಷೆ_optimize_params * params) {
    GGML_ASSERT(params && params->ಕೂಡಿಸು_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_ಕೂಡಿಸು_alloc_deps(params->user_ದತ್ತಾಂಶ, params->ಕೂಡಿಸು_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ನಕ್ಷೆ_optimize(ctx, cgraph);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_ಗಣ_n_cb(ggml_backend_t backend, ಪೂರ್ಣಾಂಕ n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ಗಣ_n_cb(ctx, n_cb);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_metal_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ggml_backend_metal_event_ದಾಖಲೆ,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_ನಕ್ಷೆ_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_metal_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    ಹಿಂತಿರುಗಿ &guid;
}

ggml_backend_t ggml_backend_metal_init(ಶೂನ್ಯ) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ಆದರೆ (ctx == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ggml_backend_t backend = (ggml_backend_t) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_ಗಣ_n_cb(backend, 1);

    ಹಿಂತಿರುಗಿ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_metal(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

ಶೂನ್ಯ ggml_backend_metal_ಗಣ_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_ಗಣ_abort_callback(ctx, abort_callback, user_ದತ್ತಾಂಶ);
}

ತಾರ್ಕಿಕ ggml_backend_metal_supports_family(ggml_backend_t backend, ಪೂರ್ಣಾಂಕ family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ಹಿಂತಿರುಗಿ ggml_metal_supports_family(ctx, family);
}

ಶೂನ್ಯ ggml_backend_metal_capture_ಮುಂದಿನ_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_ಮುಂದಿನ_compute(ctx);
}

// backend device

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಸ್ಥಿರಾಂಕ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ಹಿಂತಿರುಗಿ props_dev->name;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಹಿಂತಿರುಗಿ ggml_metal_device_get_props(ctx_dev)->desc;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_ಸ್ಮರಣೆ(ctx_dev, ಬಿಡುಗಡೆ, ಒಟ್ಟು);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);

    props->caps = {
        /* .async                = */ ನಿಜ,
        /* .host_buffer          = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr = */ ನಿಜ,
        /* .events               = */ ನಿಜ,
        /* .mmap_support         = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    ಆದರೆ (ctx == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ggml_backend_t backend = (ggml_backend_t) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_ಗಣ_n_cb(backend, 1);

    ಹಿಂತಿರುಗಿ backend;

    GGML_ಬಳಸದ(params);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_metal_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಸ್ಥಿರಾಂಕ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ಹಿಂತಿರುಗಿ props_dev->use_shared_buffers ? ggml_backend_metal_ಬಫರ್_type_shared(props_dev->device) : ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ(props_dev->device);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_metal_device_ಬಫರ್_mapped(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_ಬಫರ್_t res = ggml_metal_ಬಫರ್_map(ctx_dev, ptr, size, max_tensor_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(ggml_backend_metal_ಬಫರ್_type_mapped(props_dev->device), ggml_backend_metal_ಬಫರ್_shared_i, res, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಹಿಂತಿರುಗಿ ggml_metal_device_supports_op(ctx_dev, op);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_ಬಫರ್_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_ಬಫರ್_type_ಖಾಸಗಿ_get_name ||
        buft->iface.get_name == ggml_backend_metal_ಬಫರ್_type_mapped_get_name);

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ int64_t get_op_batch_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಹಿಂತಿರುಗಿ op->ne[1];
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            ಹಿಂತಿರುಗಿ op->ne[2];
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ggml_nrows(op);
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಹಿಂತಿರುಗಿ (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_ಗಾತ್ರ(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_ಗಾತ್ರ;
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = ಹೊಸ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    ಹಿಂತಿರುಗಿ ev;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    ಅಳಿಸು event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

ಸ್ಥಿರ ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_ಬಫರ್_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

ರಚನೆ ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_backend_metal_reg * ggml_backend_metal_reg_t;

ಸ್ಥಿರ ggml_backend_metal_reg_t ggml_backend_metal_reg_init(ಶೂನ್ಯ) {
    ggml_backend_metal_reg_t ctx = ಹೊಸ ರಚನೆ ggml_backend_metal_reg;

    ಹಿಂತಿರುಗಿ ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    ಅಳಿಸು ctx;
}

ರಚನೆ ggml_backend_metal_reg_deleter {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::unique_ptr<ರಚನೆ ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ GGML_METAL_NAME;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ size_t ggml_backend_metal_reg_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    ಹಿಂತಿರುಗಿ ctx->devices.size();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < ctx->devices.size());
    ಹಿಂತಿರುಗಿ ctx->devices[ಸೂಚ್ಯಂಕ];
}

ಸ್ಥಿರ ggml_backend_feature g_ggml_backend_metal_features[] = {
#ಆದರೆ defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#ಆದರೆ_ಅಂತ್ಯ
    { ಖಾಲಿ, ಖಾಲಿ },
};

ಸ್ಥಿರ ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    ಹಿಂತಿರುಗಿ g_ggml_backend_metal_features;

    GGML_ಬಳಸದ(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_tuning_ಗಣ_fa_vec_override(ಪೂರ್ಣಾಂಕ Q, ಪೂರ್ಣಾಂಕ NE) {
    ggml_metal_tuning::fa_vec_ಗಣ_override({ (int8_t) Q, (int8_t) NE });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_tuning_clear_fa_vec_override(ಶೂನ್ಯ) {
    ggml_metal_tuning::fa_vec_clear_override();
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_metal_tuning_fa_vec_ne11_ಪಾತ್ರೆ(int64_t ne11) {
    ಹಿಂತಿರುಗಿ ggml_metal_tuning::fa_vec_ne11_ಪಾತ್ರೆ(ne11);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_metal_tuning_fa_vec_ne01_ಪಾತ್ರೆ(int64_t ne01) {
    ಹಿಂತಿರುಗಿ ggml_metal_tuning::fa_vec_ne01_ಪಾತ್ರೆ(ne01);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_metal_tuning_fa_vec_baseline_ne(ಪೂರ್ಣಾಂಕ dk, ಪೂರ್ಣಾಂಕ dv) {
    ಹಿಂತಿರುಗಿ ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ಹಿಂತಿರುಗಿ ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ * ggml_backend_fusion_t;

ಸ್ಥಿರ ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    ಹಿಂತಿರುಗಿ ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((ರಚನೆ ggml_metal_fusion_info *) finfo);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((ರಚನೆ ggml_metal_fusion_info *) finfo);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ ** labels, uint64_t * counts, ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ ggml_metal_fusion_info_stats_get((ರಚನೆ ggml_metal_fusion_info *) finfo, labels, counts, n);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_fusion_ಗಣ_enabled(ggml_backend_fusion_t finfo, ತಾರ್ಕಿಕ enabled) {
    ggml_metal_fusion_info_ಗಣ_enabled((ರಚನೆ ggml_metal_fusion_info *) finfo, enabled);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_metal_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (strcmp(name, "ggml_backend_get_features") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_get_features;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_ಗಣ_fa_vec_override;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_fa_vec_ne11_ಪಾತ್ರೆ;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_fa_vec_ne01_ಪಾತ್ರೆ;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    ಆದರೆ (strcmp(name, "ggml_backend_fusion_get") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_get;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_stats_init;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_stats_reset;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_stats_get;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_metal_fusion_ಗಣ_enabled;
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_ವಿಳಾಸ,
};

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, ಪೂರ್ಣಾಂಕ device) {
    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    ಅಳಿಸು dev;
}

ರಚನೆ ggml_backend_device_deleter {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_reg reg;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;

    {
        ಸ್ಥಿರ std::mutex mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * env = getenv("GGML_METAL_DEVICES");
        ಆದರೆ (env) {
            g_devices = atoi(env);
        }

        ಸ್ಥಿರ std::vector<ggml_backend_device_ptr> devs;

        ಆದರೆ (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", ನಿಜ);

            ಸ್ಥಿರ ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < g_devices; ++i) {
                ಸ್ವಯಂಚಾಲಿತ * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
