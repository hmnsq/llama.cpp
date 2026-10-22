#ਅਨ੍ਤਰ੍ਭਾਵ "backend-dispatched.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend-virgl-apir.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

uint32_t backend_device_get_device_ਗਿਣਤੀ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    int32_t dev_ਗਿਣਤੀ = reg->iface.get_device_ਗਿਣਤੀ(reg);
    apir_encode_int32_t(enc, &dev_ਗਿਣਤੀ);

    ਵਾਪਸ 0;
}

uint32_t backend_device_get_ਗਿਣਤੀ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    int32_t dev_ਗਿਣਤੀ = reg->iface.get_device_ਗਿਣਤੀ(reg);
    apir_encode_int32_t(enc, &dev_ਗਿਣਤੀ);

    ਵਾਪਸ 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    ਅਚਲ ਅੱਖਰ * string = dev->iface.get_name(dev);

    ਅਚਲ size_t string_ਮਾਪ = ਲੜੀ_ਲੰਬਾਈ(string) + 1;
    apir_encode_array_ਮਾਪ(enc, string_ਮਾਪ);
    apir_encode_char_array(enc, string, string_ਮਾਪ);

    ਵਾਪਸ 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    ਅਚਲ ਅੱਖਰ * string = dev->iface.get_description(dev);

    ਅਚਲ size_t string_ਮਾਪ = ਲੜੀ_ਲੰਬਾਈ(string) + 1;
    apir_encode_array_ਮਾਪ(enc, string_ਮਾਪ);
    apir_encode_char_array(enc, string, string_ਮਾਪ);

    ਵਾਪਸ 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    ਵਾਪਸ 0;
}

uint32_t backend_device_get_ਸਮ੍ਰਿਤੀ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    size_t ਮੁਕਤ, ਕੁੱਲ;
    dev->iface.get_ਸਮ੍ਰਿਤੀ(dev, &ਮੁਕਤ, &ਕੁੱਲ);

    apir_encode_ਮਾਪ_t(enc, &ਮੁਕਤ);
    apir_encode_ਮਾਪ_t(enc, &ਕੁੱਲ);

    ਵਾਪਸ 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);

    ਅਚਲ ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    ਤਾਰਕਿਕ supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    ਵਾਪਸ 0;
}

uint32_t backend_device_get_ਬਫਰ_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    ggml_backend_ਬਫਰ_type_t bufft = dev->iface.get_ਬਫਰ_type(dev);

    apir_encode_ggml_ਬਫਰ_type(enc, bufft);

    ਵਾਪਸ 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_ਬਫਰ);
    apir_encode_bool_t(enc, &props.caps.ਬਫਰ_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    ਵਾਪਸ 0;
}

uint32_t backend_device_ਬਫਰ_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ਖਾਲੀ * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ਜੇ (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(dec);
        ਵਾਪਸ 1;
    }

    size_t size;
    apir_decode_ਮਾਪ_t(dec, &size);
    size_t max_tensor_ਮਾਪ;
    apir_decode_ਮਾਪ_t(dec, &max_tensor_ਮਾਪ);

    ggml_backend_ਬਫਰ_t ਬਫਰ;
    ਬਫਰ = dev->iface.ਬਫਰ_from_host_ptr(dev, shmem_ptr, size, max_tensor_ਮਾਪ);

    apir_encode_ggml_ਬਫਰ(enc, ਬਫਰ);
    apir_encode_ggml_ਬਫਰ_type(enc, ਬਫਰ->buft);

    ਜੇ (ਬਫਰ) {
        apir_track_backend_ਬਫਰ(ਬਫਰ);
    }

    ਵਾਪਸ 0;
}
