#अन्तर्भाव "backend-dispatched.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdint>

uint32_t backend_device_get_device_मोजणी(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    int32_t dev_मोजणी = reg->iface.get_device_मोजणी(reg);
    apir_encode_int32_t(enc, &dev_मोजणी);

    परत 0;
}

uint32_t backend_device_get_मोजणी(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    int32_t dev_मोजणी = reg->iface.get_device_मोजणी(reg);
    apir_encode_int32_t(enc, &dev_मोजणी);

    परत 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    स्थिरांक अक्षर * string = dev->iface.get_name(dev);

    स्थिरांक माप_प्रकार string_size = म_माप(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    परत 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    स्थिरांक अक्षर * string = dev->iface.get_description(dev);

    स्थिरांक माप_प्रकार string_size = म_माप(string) + 1;
    apir_encode_array_size(enc, string_size);
    apir_encode_char_array(enc, string, string_size);

    परत 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    परत 0;
}

uint32_t backend_device_get_memory(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    माप_प्रकार मुक्त, total;
    dev->iface.get_memory(dev, &मुक्त, &total);

    apir_encode_size_t(enc, &मुक्त);
    apir_encode_size_t(enc, &total);

    परत 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    स्थिरांक ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    तार्किक supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    परत 0;
}

uint32_t backend_device_get_buffer_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    ggml_backend_buffer_type_t bufft = dev->iface.get_buffer_type(dev);

    apir_encode_ggml_buffer_type(enc, bufft);

    परत 0;
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

    परत 0;
}

uint32_t backend_device_buffer_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    शून्य * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    जर (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        परत 1;
    }

    माप_प्रकार size;
    apir_decode_size_t(dec, &size);
    माप_प्रकार max_tensor_size;
    apir_decode_size_t(dec, &max_tensor_size);

    ggml_backend_buffer_t buffer;
    buffer = dev->iface.buffer_from_host_ptr(dev, shmem_ptr, size, max_tensor_size);

    apir_encode_ggml_buffer(enc, buffer);
    apir_encode_ggml_buffer_type(enc, buffer->buft);

    जर (buffer) {
        apir_track_backend_buffer(buffer);
    }

    परत 0;
}
