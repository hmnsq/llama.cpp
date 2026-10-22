#ضمن "backend-dispatched.h"
#ضمن "backend-virgl-apir.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"

#ضمن <cstdint>

ساكن uint32_t validate_بفر_operation(size_t وٿي, size_t size, ثابت حرف * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    إذا (وٿي > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, وٿي, size);
        أرجع 1;
    }

    أرجع 0;  // Valid
}

uint32_t backend_بفر_get_بنياد(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_اڻ_استعمال_ٿيل(ctx);
    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    إذا (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    uintptr_t بنياد = (uintptr_t) بفر->iface.get_بنياد(بفر);
    apir_encode_uintptr_t(enc, &بنياد);

    أرجع 0;
}

uint32_t backend_بفر_سيٽ_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_اڻ_استعمال_ٿيل(ctx);
    GGML_اڻ_استعمال_ٿيل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    إذا (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t وٿي;
    apir_decode_سائيز_t(dec, &وٿي);

    size_t size;
    apir_decode_سائيز_t(dec, &size);

    إذا (validate_بفر_operation(وٿي, size, __func__) != 0) {
        أرجع 1;
    }

    فراغ * shmem_معلومات = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    إذا (!shmem_معلومات) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        أرجع 1;
    }

    بفر->iface.سيٽ_tensor(بفر, tensor, shmem_معلومات, وٿي, size);

    أرجع 0;
}

uint32_t backend_بفر_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_اڻ_استعمال_ٿيل(ctx);
    GGML_اڻ_استعمال_ٿيل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    إذا (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    ثابت ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t وٿي;
    apir_decode_سائيز_t(dec, &وٿي);

    size_t size;
    apir_decode_سائيز_t(dec, &size);

    إذا (validate_بفر_operation(وٿي, size, __func__) != 0) {
        أرجع 1;
    }

    فراغ * shmem_معلومات = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    إذا (!shmem_معلومات) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        أرجع 1;
    }

    بفر->iface.get_tensor(بفر, tensor, shmem_معلومات, وٿي, size);

    أرجع 0;
}

uint32_t backend_بفر_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_اڻ_استعمال_ٿيل(ctx);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    إذا (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    ثابت ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    منطقي ret = بفر->iface.cpy_tensor(بفر, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    أرجع 0;
}

uint32_t backend_بفر_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_اڻ_استعمال_ٿيل(ctx);
    GGML_اڻ_استعمال_ٿيل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    إذا (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    بفر->iface.clear(بفر, value);

    أرجع 0;
}

uint32_t backend_بفر_free_بفر(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_اڻ_استعمال_ٿيل(ctx);
    GGML_اڻ_استعمال_ٿيل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    إذا (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    إذا (!apir_untrack_backend_بفر(بفر)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (فراغ *) بفر);
        أرجع 1;
    }

    بفر->iface.free_بفر(بفر);

    أرجع 0;
}
