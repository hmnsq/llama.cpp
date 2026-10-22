#అన్తర్భావ "backend-dispatched.h"
#అన్తర్భావ "backend-virgl-apir.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstdint>

స్థిర uint32_t validate_బఫరు_operation(size_t స్థానభ్రంశం, size_t size, స్థిరరాశి అక్షరం * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    ఒకవేళ (స్థానభ్రంశం > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, స్థానభ్రంశం, size);
        తిరిగి 1;
    }

    తిరిగి 0;  // Valid
}

uint32_t backend_బఫరు_get_పునాది(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    ggml_backend_బఫరు_t బఫరు;
    బఫరు = apir_decode_ggml_బఫరు(dec);

    ఒకవేళ (!బఫరు || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        తిరిగి 1;
    }

    uintptr_t పునాది = (uintptr_t) బఫరు->iface.get_పునాది(బఫరు);
    apir_encode_uintptr_t(enc, &పునాది);

    తిరిగి 0;
}

uint32_t backend_బఫరు_సమితి_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(enc);

    ggml_backend_బఫరు_t బఫరు;
    బఫరు = apir_decode_ggml_బఫరు(dec);

    ఒకవేళ (!బఫరు || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        తిరిగి 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t స్థానభ్రంశం;
    apir_decode_కొలత_t(dec, &స్థానభ్రంశం);

    size_t size;
    apir_decode_కొలత_t(dec, &size);

    ఒకవేళ (validate_బఫరు_operation(స్థానభ్రంశం, size, __func__) != 0) {
        తిరిగి 1;
    }

    శూన్య * shmem_దత్తాంశం = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    ఒకవేళ (!shmem_దత్తాంశం) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        తిరిగి 1;
    }

    బఫరు->iface.సమితి_tensor(బఫరు, tensor, shmem_దత్తాంశం, స్థానభ్రంశం, size);

    తిరిగి 0;
}

uint32_t backend_బఫరు_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(enc);

    ggml_backend_బఫరు_t బఫరు;
    బఫరు = apir_decode_ggml_బఫరు(dec);

    ఒకవేళ (!బఫరు || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        తిరిగి 1;
    }

    స్థిరరాశి ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t స్థానభ్రంశం;
    apir_decode_కొలత_t(dec, &స్థానభ్రంశం);

    size_t size;
    apir_decode_కొలత_t(dec, &size);

    ఒకవేళ (validate_బఫరు_operation(స్థానభ్రంశం, size, __func__) != 0) {
        తిరిగి 1;
    }

    శూన్య * shmem_దత్తాంశం = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ఒకవేళ (!shmem_దత్తాంశం) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        తిరిగి 1;
    }

    బఫరు->iface.get_tensor(బఫరు, tensor, shmem_దత్తాంశం, స్థానభ్రంశం, size);

    తిరిగి 0;
}

uint32_t backend_బఫరు_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);

    ggml_backend_బఫరు_t బఫరు;
    బఫరు = apir_decode_ggml_బఫరు(dec);

    ఒకవేళ (!బఫరు || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        తిరిగి 1;
    }

    స్థిరరాశి ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    తార్కిక ret = బఫరు->iface.cpy_tensor(బఫరు, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    తిరిగి 0;
}

uint32_t backend_బఫరు_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(enc);

    ggml_backend_బఫరు_t బఫరు;
    బఫరు = apir_decode_ggml_బఫరు(dec);

    ఒకవేళ (!బఫరు || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        తిరిగి 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    బఫరు->iface.clear(బఫరు, value);

    తిరిగి 0;
}

uint32_t backend_బఫరు_free_బఫరు(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);
    GGML_వాడని(enc);

    ggml_backend_బఫరు_t బఫరు;
    బఫరు = apir_decode_ggml_బఫరు(dec);

    ఒకవేళ (!బఫరు || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        తిరిగి 1;
    }

    ఒకవేళ (!apir_untrack_backend_బఫరు(బఫరు)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (శూన్య *) బఫరు);
        తిరిగి 1;
    }

    బఫరు->iface.free_బఫరు(బఫరు);

    తిరిగి 0;
}
