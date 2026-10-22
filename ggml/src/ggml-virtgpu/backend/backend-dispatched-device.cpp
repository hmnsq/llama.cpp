#অন্তর্ভাব "backend-dispatched.h"
#অন্তর্ভাব "backend-virgl-apir.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstdint>

uint32_t backend_device_get_device_গুনতি(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    int32_t dev_গুনতি = reg->iface.get_device_গুনতি(reg);
    apir_encode_int32_t(enc, &dev_গুনতি);

    ফেরত 0;
}

uint32_t backend_device_get_গুনতি(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    int32_t dev_গুনতি = reg->iface.get_device_গুনতি(reg);
    apir_encode_int32_t(enc, &dev_গুনতি);

    ফেরত 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ধ্রুবক অক্ষর * string = dev->iface.get_name(dev);

    ধ্রুবক size_t string_আকার = strlen(string) + 1;
    apir_encode_array_আকার(enc, string_আকার);
    apir_encode_char_array(enc, string, string_আকার);

    ফেরত 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ধ্রুবক অক্ষর * string = dev->iface.get_description(dev);

    ধ্রুবক size_t string_আকার = strlen(string) + 1;
    apir_encode_array_আকার(enc, string_আকার);
    apir_encode_char_array(enc, string, string_আকার);

    ফেরত 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    ফেরত 0;
}

uint32_t backend_device_get_memory(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    size_t মুক্ত, যোগ;
    dev->iface.get_memory(dev, &মুক্ত, &যোগ);

    apir_encode_আকার_t(enc, &মুক্ত);
    apir_encode_আকার_t(enc, &যোগ);

    ফেরত 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    ধ্রুবক ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    যৌক্তিক supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    ফেরত 0;
}

uint32_t backend_device_get_বাফার_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ggml_backend_বাফার_type_t bufft = dev->iface.get_বাফার_type(dev);

    apir_encode_ggml_বাফার_type(enc, bufft);

    ফেরত 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_বাফার);
    apir_encode_bool_t(enc, &props.caps.বাফার_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    ফেরত 0;
}

uint32_t backend_device_বাফার_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    শূন্য * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    যদি (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        ফেরত 1;
    }

    size_t size;
    apir_decode_আকার_t(dec, &size);
    size_t max_tensor_আকার;
    apir_decode_আকার_t(dec, &max_tensor_আকার);

    ggml_backend_বাফার_t বাফার;
    বাফার = dev->iface.বাফার_from_host_ptr(dev, shmem_ptr, size, max_tensor_আকার);

    apir_encode_ggml_বাফার(enc, বাফার);
    apir_encode_ggml_বাফার_type(enc, বাফার->buft);

    যদি (বাফার) {
        apir_track_backend_বাফার(বাফার);
    }

    ফেরত 0;
}
