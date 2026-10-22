#अन्तर्भाव "backend-dispatched.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdint>

स्थिर uint32_t validate_बफर_operation(size_t विस्थापनम्, size_t size, ध्रुव वर्ण * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    यदि (विस्थापनम् > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, विस्थापनम्, size);
        प्रत्यागम 1;
    }

    प्रत्यागम 0;  // Valid
}

uint32_t backend_बफर_get_आधारः(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(dec);

    यदि (!बफर || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        प्रत्यागम 1;
    }

    uintptr_t आधारः = (uintptr_t) बफर->iface.get_आधारः(बफर);
    apir_encode_uintptr_t(enc, &आधारः);

    प्रत्यागम 0;
}

uint32_t backend_बफर_समुच्चयः_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(dec);

    यदि (!बफर || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        प्रत्यागम 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t विस्थापनम्;
    apir_decode_आकार_t(dec, &विस्थापनम्);

    size_t size;
    apir_decode_आकार_t(dec, &size);

    यदि (validate_बफर_operation(विस्थापनम्, size, __func__) != 0) {
        प्रत्यागम 1;
    }

    शून्य * shmem_दत्त = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    यदि (!shmem_दत्त) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        प्रत्यागम 1;
    }

    बफर->iface.समुच्चयः_tensor(बफर, tensor, shmem_दत्त, विस्थापनम्, size);

    प्रत्यागम 0;
}

uint32_t backend_बफर_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(dec);

    यदि (!बफर || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        प्रत्यागम 1;
    }

    ध्रुव ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t विस्थापनम्;
    apir_decode_आकार_t(dec, &विस्थापनम्);

    size_t size;
    apir_decode_आकार_t(dec, &size);

    यदि (validate_बफर_operation(विस्थापनम्, size, __func__) != 0) {
        प्रत्यागम 1;
    }

    शून्य * shmem_दत्त = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    यदि (!shmem_दत्त) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        प्रत्यागम 1;
    }

    बफर->iface.get_tensor(बफर, tensor, shmem_दत्त, विस्थापनम्, size);

    प्रत्यागम 0;
}

uint32_t backend_बफर_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(dec);

    यदि (!बफर || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        प्रत्यागम 1;
    }

    ध्रुव ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    तार्किक ret = बफर->iface.cpy_tensor(बफर, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    प्रत्यागम 0;
}

uint32_t backend_बफर_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(dec);

    यदि (!बफर || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        प्रत्यागम 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    बफर->iface.clear(बफर, value);

    प्रत्यागम 0;
}

uint32_t backend_बफर_free_बफर(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(dec);

    यदि (!बफर || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        प्रत्यागम 1;
    }

    यदि (!apir_untrack_backend_बफर(बफर)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (शून्य *) बफर);
        प्रत्यागम 1;
    }

    बफर->iface.free_बफर(बफर);

    प्रत्यागम 0;
}
