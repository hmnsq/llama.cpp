#ضمن "backend-dispatched.h"
#ضمن "backend-virgl-apir.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"

#ضمن <cstdint>

ساكن uint32_t validate_میانگیر_operation(size_t جابه_جایی, size_t size, ثابت حرف * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    إذا (جابه_جایی > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, جابه_جایی, size);
        أرجع 1;
    }

    أرجع 0;  // Valid
}

uint32_t backend_میانگیر_get_پایه(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    ggml_backend_میانگیر_t میانگیر;
    میانگیر = apir_decode_ggml_میانگیر(dec);

    إذا (!میانگیر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    uintptr_t پایه = (uintptr_t) میانگیر->iface.get_پایه(میانگیر);
    apir_encode_uintptr_t(enc, &پایه);

    أرجع 0;
}

uint32_t backend_میانگیر_مجموعه_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(enc);

    ggml_backend_میانگیر_t میانگیر;
    میانگیر = apir_decode_ggml_میانگیر(dec);

    إذا (!میانگیر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t جابه_جایی;
    apir_decode_اندازه_t(dec, &جابه_جایی);

    size_t size;
    apir_decode_اندازه_t(dec, &size);

    إذا (validate_میانگیر_operation(جابه_جایی, size, __func__) != 0) {
        أرجع 1;
    }

    فراغ * shmem_داده = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    إذا (!shmem_داده) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        أرجع 1;
    }

    میانگیر->iface.مجموعه_tensor(میانگیر, tensor, shmem_داده, جابه_جایی, size);

    أرجع 0;
}

uint32_t backend_میانگیر_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(enc);

    ggml_backend_میانگیر_t میانگیر;
    میانگیر = apir_decode_ggml_میانگیر(dec);

    إذا (!میانگیر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    ثابت ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t جابه_جایی;
    apir_decode_اندازه_t(dec, &جابه_جایی);

    size_t size;
    apir_decode_اندازه_t(dec, &size);

    إذا (validate_میانگیر_operation(جابه_جایی, size, __func__) != 0) {
        أرجع 1;
    }

    فراغ * shmem_داده = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    إذا (!shmem_داده) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        أرجع 1;
    }

    میانگیر->iface.get_tensor(میانگیر, tensor, shmem_داده, جابه_جایی, size);

    أرجع 0;
}

uint32_t backend_میانگیر_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);

    ggml_backend_میانگیر_t میانگیر;
    میانگیر = apir_decode_ggml_میانگیر(dec);

    إذا (!میانگیر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    ثابت ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    منطقي ret = میانگیر->iface.cpy_tensor(میانگیر, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    أرجع 0;
}

uint32_t backend_میانگیر_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(enc);

    ggml_backend_میانگیر_t میانگیر;
    میانگیر = apir_decode_ggml_میانگیر(dec);

    إذا (!میانگیر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    میانگیر->iface.clear(میانگیر, value);

    أرجع 0;
}

uint32_t backend_میانگیر_free_میانگیر(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_استفادهنشده(ctx);
    GGML_استفادهنشده(enc);

    ggml_backend_میانگیر_t میانگیر;
    میانگیر = apir_decode_ggml_میانگیر(dec);

    إذا (!میانگیر || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        أرجع 1;
    }

    إذا (!apir_untrack_backend_میانگیر(میانگیر)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (فراغ *) میانگیر);
        أرجع 1;
    }

    میانگیر->iface.free_میانگیر(میانگیر);

    أرجع 0;
}
