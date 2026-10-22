#شامل "backend-dispatched.h"
#شامل "backend-virgl-apir.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"

#شامل <cstdint>

ساکن uint32_t validate_بفر_operation(size_t انحراف, size_t size, ثابت حرف * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    اگر (انحراف > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, انحراف, size);
        واپس 1;
    }

    واپس 0;  // Valid
}

uint32_t backend_بفر_get_بنیاد(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    اگر (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        واپس 1;
    }

    uintptr_t بنیاد = (uintptr_t) بفر->iface.get_بنیاد(بفر);
    apir_encode_uintptr_t(enc, &بنیاد);

    واپس 0;
}

uint32_t backend_بفر_سیٹ_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    اگر (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        واپس 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t انحراف;
    apir_decode_سائز_t(dec, &انحراف);

    size_t size;
    apir_decode_سائز_t(dec, &size);

    اگر (validate_بفر_operation(انحراف, size, __func__) != 0) {
        واپس 1;
    }

    خالی * shmem_ڈیٹا = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    اگر (!shmem_ڈیٹا) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        واپس 1;
    }

    بفر->iface.سیٹ_tensor(بفر, tensor, shmem_ڈیٹا, انحراف, size);

    واپس 0;
}

uint32_t backend_بفر_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    اگر (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        واپس 1;
    }

    ثابت ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t انحراف;
    apir_decode_سائز_t(dec, &انحراف);

    size_t size;
    apir_decode_سائز_t(dec, &size);

    اگر (validate_بفر_operation(انحراف, size, __func__) != 0) {
        واپس 1;
    }

    خالی * shmem_ڈیٹا = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    اگر (!shmem_ڈیٹا) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        واپس 1;
    }

    بفر->iface.get_tensor(بفر, tensor, shmem_ڈیٹا, انحراف, size);

    واپس 0;
}

uint32_t backend_بفر_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    اگر (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        واپس 1;
    }

    ثابت ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    منطقی ret = بفر->iface.cpy_tensor(بفر, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    واپس 0;
}

uint32_t backend_بفر_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    اگر (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        واپس 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    بفر->iface.clear(بفر, value);

    واپس 0;
}

uint32_t backend_بفر_free_بفر(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(enc);

    ggml_backend_بفر_t بفر;
    بفر = apir_decode_ggml_بفر(dec);

    اگر (!بفر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        واپس 1;
    }

    اگر (!apir_untrack_backend_بفر(بفر)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (خالی *) بفر);
        واپس 1;
    }

    بفر->iface.free_بفر(بفر);

    واپس 0;
}
