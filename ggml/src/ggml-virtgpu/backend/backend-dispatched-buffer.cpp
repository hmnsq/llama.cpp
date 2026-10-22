#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdint>

ಸ್ಥಿರ uint32_t validate_ಬಫರ್_operation(size_t ಸ್ಥಾನಾಂತರ, size_t size, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    ಆದರೆ (ಸ್ಥಾನಾಂತರ > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, ಸ್ಥಾನಾಂತರ, size);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಹಿಂತಿರುಗಿ 0;  // Valid
}

uint32_t backend_ಬಫರ್_get_ತಳ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    ggml_backend_ಬಫರ್_t ಬಫರ್;
    ಬಫರ್ = apir_decode_ggml_ಬಫರ್(dec);

    ಆದರೆ (!ಬಫರ್ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    uintptr_t ತಳ = (uintptr_t) ಬಫರ್->iface.get_ತಳ(ಬಫರ್);
    apir_encode_uintptr_t(enc, &ತಳ);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_ಗಣ_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(enc);

    ggml_backend_ಬಫರ್_t ಬಫರ್;
    ಬಫರ್ = apir_decode_ggml_ಬಫರ್(dec);

    ಆದರೆ (!ಬಫರ್ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t ಸ್ಥಾನಾಂತರ;
    apir_decode_ಗಾತ್ರ_t(dec, &ಸ್ಥಾನಾಂತರ);

    size_t size;
    apir_decode_ಗಾತ್ರ_t(dec, &size);

    ಆದರೆ (validate_ಬಫರ್_operation(ಸ್ಥಾನಾಂತರ, size, __func__) != 0) {
        ಹಿಂತಿರುಗಿ 1;
    }

    ಶೂನ್ಯ * shmem_ದತ್ತಾಂಶ = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    ಆದರೆ (!shmem_ದತ್ತಾಂಶ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಬಫರ್->iface.ಗಣ_tensor(ಬಫರ್, tensor, shmem_ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(enc);

    ggml_backend_ಬಫರ್_t ಬಫರ್;
    ಬಫರ್ = apir_decode_ggml_ಬಫರ್(dec);

    ಆದರೆ (!ಬಫರ್ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t ಸ್ಥಾನಾಂತರ;
    apir_decode_ಗಾತ್ರ_t(dec, &ಸ್ಥಾನಾಂತರ);

    size_t size;
    apir_decode_ಗಾತ್ರ_t(dec, &size);

    ಆದರೆ (validate_ಬಫರ್_operation(ಸ್ಥಾನಾಂತರ, size, __func__) != 0) {
        ಹಿಂತಿರುಗಿ 1;
    }

    ಶೂನ್ಯ * shmem_ದತ್ತಾಂಶ = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ಆದರೆ (!shmem_ದತ್ತಾಂಶ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಬಫರ್->iface.get_tensor(ಬಫರ್, tensor, shmem_ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);

    ggml_backend_ಬಫರ್_t ಬಫರ್;
    ಬಫರ್ = apir_decode_ggml_ಬಫರ್(dec);

    ಆದರೆ (!ಬಫರ್ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    ತಾರ್ಕಿಕ ret = ಬಫರ್->iface.cpy_tensor(ಬಫರ್, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(enc);

    ggml_backend_ಬಫರ್_t ಬಫರ್;
    ಬಫರ್ = apir_decode_ggml_ಬಫರ್(dec);

    ಆದರೆ (!ಬಫರ್ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    ಬಫರ್->iface.clear(ಬಫರ್, value);

    ಹಿಂತಿರುಗಿ 0;
}

uint32_t backend_ಬಫರ್_free_ಬಫರ್(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(enc);

    ggml_backend_ಬಫರ್_t ಬಫರ್;
    ಬಫರ್ = apir_decode_ggml_ಬಫರ್(dec);

    ಆದರೆ (!ಬಫರ್ || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಆದರೆ (!apir_untrack_backend_ಬಫರ್(ಬಫರ್)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (ಶೂನ್ಯ *) ಬಫರ್);
        ಹಿಂತಿರುಗಿ 1;
    }

    ಬಫರ್->iface.free_ಬಫರ್(ಬಫರ್);

    ಹಿಂತಿರುಗಿ 0;
}
