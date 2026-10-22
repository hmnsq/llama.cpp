#অন্তর্ভাৱ "backend-dispatched.h"
#অন্তর্ভাৱ "backend-virgl-apir.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-backend.h"
#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ <cstdint>

uint32_t backend_device_get_device_গুনতি(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    int32_t dev_গুনতি = reg->iface.get_device_গুনতি(reg);
    apir_encode_int32_t(enc, &dev_গুনতি);

    ঘূৰাই 0;
}

uint32_t backend_device_get_গুনতি(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    int32_t dev_গুনতি = reg->iface.get_device_গুনতি(reg);
    apir_encode_int32_t(enc, &dev_গুনতি);

    ঘূৰাই 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ধ্রুৱ ৱর্ণ * string = dev->iface.get_name(dev);

    ধ্রুৱ size_t string_size = strlen(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    ঘূৰাই 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ধ্রুৱ ৱর্ণ * string = dev->iface.get_description(dev);

    ধ্রুৱ size_t string_size = strlen(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    ঘূৰাই 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    ঘূৰাই 0;
}

uint32_t backend_device_get_memory(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    size_t মুক্ত, total;
    dev->iface.get_memory(dev, &মুক্ত, &total);

    apir_encode_size_t(enc, &মুক্ত);
    apir_encode_size_t(enc, &total);

    ঘূৰাই 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    ধ্রুৱ ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    তার্কিক supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    ঘূৰাই 0;
}

uint32_t backend_device_get_buffer_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ggml_backend_buffer_type_t bufft = dev->iface.get_buffer_type(dev);

    apir_encode_ggml_buffer_type(enc, bufft);

    ঘূৰাই 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_buffer);
    apir_encode_bool_t(enc, &props.caps.buffer_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    ঘূৰাই 0;
}

uint32_t backend_device_buffer_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    শূন্য * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    যদি (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        ঘূৰাই 1;
    }

    size_t size;
    apir_decode_size_t(dec, &size);
    size_t max_tensor_size;
    apir_decode_size_t(dec, &max_tensor_size);

    ggml_backend_buffer_t buffer;
    buffer = dev->iface.buffer_from_host_ptr(dev, shmem_ptr, size, max_tensor_size);

    apir_encode_ggml_buffer(enc, buffer);
    apir_encode_ggml_buffer_type(enc, buffer->buft);

    যদি (buffer) {
        apir_track_backend_buffer(buffer);
    }

    ঘূৰাই 0;
}
