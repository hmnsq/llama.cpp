#ਅਨ੍ਤਰ੍ਭਾਵ "backend-dispatched.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend-virgl-apir.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

ਸਥਿਰ uint32_t validate_ਬਫਰ_operation(size_t ਵਿੱਥ, size_t size, ਅਚਲ ਅੱਖਰ * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    ਜੇ (ਵਿੱਥ > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, ਵਿੱਥ, size);
        ਵਾਪਸ 1;
    }

    ਵਾਪਸ 0;  // Valid
}

uint32_t backend_ਬਫਰ_get_ਅਧਾਰ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    ggml_backend_ਬਫਰ_t ਬਫਰ;
    ਬਫਰ = apir_decode_ggml_ਬਫਰ(dec);

    ਜੇ (!ਬਫਰ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ਵਾਪਸ 1;
    }

    uintptr_t ਅਧਾਰ = (uintptr_t) ਬਫਰ->iface.get_ਅਧਾਰ(ਬਫਰ);
    apir_encode_uintptr_t(enc, &ਅਧਾਰ);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_ਸੈੱਟ_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(enc);

    ggml_backend_ਬਫਰ_t ਬਫਰ;
    ਬਫਰ = apir_decode_ggml_ਬਫਰ(dec);

    ਜੇ (!ਬਫਰ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ਵਾਪਸ 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t ਵਿੱਥ;
    apir_decode_ਮਾਪ_t(dec, &ਵਿੱਥ);

    size_t size;
    apir_decode_ਮਾਪ_t(dec, &size);

    ਜੇ (validate_ਬਫਰ_operation(ਵਿੱਥ, size, __func__) != 0) {
        ਵਾਪਸ 1;
    }

    ਖਾਲੀ * shmem_ਅੰਕੜੇ = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    ਜੇ (!shmem_ਅੰਕੜੇ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        ਵਾਪਸ 1;
    }

    ਬਫਰ->iface.ਸੈੱਟ_tensor(ਬਫਰ, tensor, shmem_ਅੰਕੜੇ, ਵਿੱਥ, size);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(enc);

    ggml_backend_ਬਫਰ_t ਬਫਰ;
    ਬਫਰ = apir_decode_ggml_ਬਫਰ(dec);

    ਜੇ (!ਬਫਰ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ਵਾਪਸ 1;
    }

    ਅਚਲ ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t ਵਿੱਥ;
    apir_decode_ਮਾਪ_t(dec, &ਵਿੱਥ);

    size_t size;
    apir_decode_ਮਾਪ_t(dec, &size);

    ਜੇ (validate_ਬਫਰ_operation(ਵਿੱਥ, size, __func__) != 0) {
        ਵਾਪਸ 1;
    }

    ਖਾਲੀ * shmem_ਅੰਕੜੇ = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ਜੇ (!shmem_ਅੰਕੜੇ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        ਵਾਪਸ 1;
    }

    ਬਫਰ->iface.get_tensor(ਬਫਰ, tensor, shmem_ਅੰਕੜੇ, ਵਿੱਥ, size);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);

    ggml_backend_ਬਫਰ_t ਬਫਰ;
    ਬਫਰ = apir_decode_ggml_ਬਫਰ(dec);

    ਜੇ (!ਬਫਰ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ਵਾਪਸ 1;
    }

    ਅਚਲ ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    ਤਾਰਕਿਕ ret = ਬਫਰ->iface.cpy_tensor(ਬਫਰ, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(enc);

    ggml_backend_ਬਫਰ_t ਬਫਰ;
    ਬਫਰ = apir_decode_ggml_ਬਫਰ(dec);

    ਜੇ (!ਬਫਰ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ਵਾਪਸ 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    ਬਫਰ->iface.clear(ਬਫਰ, value);

    ਵਾਪਸ 0;
}

uint32_t backend_ਬਫਰ_free_ਬਫਰ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(enc);

    ggml_backend_ਬਫਰ_t ਬਫਰ;
    ਬਫਰ = apir_decode_ggml_ਬਫਰ(dec);

    ਜੇ (!ਬਫਰ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ਵਾਪਸ 1;
    }

    ਜੇ (!apir_untrack_backend_ਬਫਰ(ਬਫਰ)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (ਖਾਲੀ *) ਬਫਰ);
        ਵਾਪਸ 1;
    }

    ਬਫਰ->iface.free_ਬਫਰ(ਬਫਰ);

    ਵਾਪਸ 0;
}
