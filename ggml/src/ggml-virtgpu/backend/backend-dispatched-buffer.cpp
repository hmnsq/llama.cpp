#অন্তর্ভাব "backend-dispatched.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstdint>

স্থির uint32_t validate_বাফার_operation(size_t offset, size_t size, ধ্রুবক অক্ষর * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    যদি (offset > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, offset, size);
        ফেরত 1;
    }

    ফেরত 0;  // Valid
}

uint32_t backend_বাফার_get_base(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_বাফার_t বাফার;
    বাফার = apir_decode_ggml_বাফার(dec);

    যদি (!বাফার || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ফেরত 1;
    }

    uintptr_t base = (uintptr_t) বাফার->iface.get_base(বাফার);
    apir_encode_uintptr_t(enc, &base);

    ফেরত 0;
}

uint32_t backend_বাফার_set_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_বাফার_t বাফার;
    বাফার = apir_decode_ggml_বাফার(dec);

    যদি (!বাফার || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ফেরত 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t offset;
    apir_decode_আকার_t(dec, &offset);

    size_t size;
    apir_decode_আকার_t(dec, &size);

    যদি (validate_বাফার_operation(offset, size, __func__) != 0) {
        ফেরত 1;
    }

    শূন্য * shmem_তথ্য = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    যদি (!shmem_তথ্য) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        ফেরত 1;
    }

    বাফার->iface.set_tensor(বাফার, tensor, shmem_তথ্য, offset, size);

    ফেরত 0;
}

uint32_t backend_বাফার_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_বাফার_t বাফার;
    বাফার = apir_decode_ggml_বাফার(dec);

    যদি (!বাফার || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ফেরত 1;
    }

    ধ্রুবক ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t offset;
    apir_decode_আকার_t(dec, &offset);

    size_t size;
    apir_decode_আকার_t(dec, &size);

    যদি (validate_বাফার_operation(offset, size, __func__) != 0) {
        ফেরত 1;
    }

    শূন্য * shmem_তথ্য = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    যদি (!shmem_তথ্য) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        ফেরত 1;
    }

    বাফার->iface.get_tensor(বাফার, tensor, shmem_তথ্য, offset, size);

    ফেরত 0;
}

uint32_t backend_বাফার_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    ggml_backend_বাফার_t বাফার;
    বাফার = apir_decode_ggml_বাফার(dec);

    যদি (!বাফার || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ফেরত 1;
    }

    ধ্রুবক ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    যৌক্তিক ret = বাফার->iface.cpy_tensor(বাফার, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    ফেরত 0;
}

uint32_t backend_বাফার_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_বাফার_t বাফার;
    বাফার = apir_decode_ggml_বাফার(dec);

    যদি (!বাফার || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ফেরত 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    বাফার->iface.clear(বাফার, value);

    ফেরত 0;
}

uint32_t backend_বাফার_free_বাফার(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_বাফার_t বাফার;
    বাফার = apir_decode_ggml_বাফার(dec);

    যদি (!বাফার || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        ফেরত 1;
    }

    যদি (!apir_untrack_backend_বাফার(বাফার)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (শূন্য *) বাফার);
        ফেরত 1;
    }

    বাফার->iface.free_বাফার(বাফার);

    ফেরত 0;
}
