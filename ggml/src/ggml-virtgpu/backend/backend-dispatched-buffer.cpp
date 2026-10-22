#समावेश "backend-dispatched.h"
#समावेश "backend-virgl-apir.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"

#समावेश <cstdint>

अटल uint32_t validate_बफर_operation(माप_प्रकार offset, माप_प्रकार size, स्थिर अक्षर * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    अगर (offset > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, offset, size);
        वापस 1;
    }

    वापस 0;  // Valid
}

uint32_t backend_बफर_get_base(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(डेसी);

    अगर (!बफर || apir_decoder_get_fatal(डेसी)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        वापस 1;
    }

    uintptr_t base = (uintptr_t) बफर->iface.get_base(बफर);
    apir_encode_uintptr_t(enc, &base);

    वापस 0;
}

uint32_t backend_बफर_set_tensor(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(डेसी);

    अगर (!बफर || apir_decoder_get_fatal(डेसी)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        वापस 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(डेसी);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(डेसी, &shmem_res_id);

    माप_प्रकार offset;
    apir_decode_आकार_t(डेसी, &offset);

    माप_प्रकार size;
    apir_decode_आकार_t(डेसी, &size);

    अगर (validate_बफर_operation(offset, size, __func__) != 0) {
        वापस 1;
    }

    व्योम * shmem_तथ्य = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    अगर (!shmem_तथ्य) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        वापस 1;
    }

    बफर->iface.set_tensor(बफर, tensor, shmem_तथ्य, offset, size);

    वापस 0;
}

uint32_t backend_बफर_get_tensor(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(डेसी);

    अगर (!बफर || apir_decoder_get_fatal(डेसी)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        वापस 1;
    }

    स्थिर ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(डेसी);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(डेसी, &shmem_res_id);

    माप_प्रकार offset;
    apir_decode_आकार_t(डेसी, &offset);

    माप_प्रकार size;
    apir_decode_आकार_t(डेसी, &size);

    अगर (validate_बफर_operation(offset, size, __func__) != 0) {
        वापस 1;
    }

    व्योम * shmem_तथ्य = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    अगर (!shmem_तथ्य) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        वापस 1;
    }

    बफर->iface.get_tensor(बफर, tensor, shmem_तथ्य, offset, size);

    वापस 0;
}

uint32_t backend_बफर_cpy_tensor(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(डेसी);

    अगर (!बफर || apir_decoder_get_fatal(डेसी)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        वापस 1;
    }

    स्थिर ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(डेसी);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(डेसी);

    bool ret = बफर->iface.cpy_tensor(बफर, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    वापस 0;
}

uint32_t backend_बफर_clear(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(डेसी);

    अगर (!बफर || apir_decoder_get_fatal(डेसी)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        वापस 1;
    }

    uint8_t value;
    apir_decode_uint8_t(डेसी, &value);

    बफर->iface.सफा(बफर, value);

    वापस 0;
}

uint32_t backend_बफर_free_बफर(apir_encoder * enc, apir_decoder * डेसी, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_बफर_t बफर;
    बफर = apir_decode_ggml_बफर(डेसी);

    अगर (!बफर || apir_decoder_get_fatal(डेसी)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        वापस 1;
    }

    अगर (!apir_untrack_backend_बफर(बफर)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (व्योम *) बफर);
        वापस 1;
    }

    बफर->iface.free_बफर(बफर);

    वापस 0;
}
