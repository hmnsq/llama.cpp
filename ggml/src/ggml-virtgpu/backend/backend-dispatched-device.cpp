#شامل "backend-dispatched.h"
#شامل "backend-virgl-apir.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"

#شامل <cstdint>

uint32_t backend_device_get_device_گنتی(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    int32_t dev_گنتی = reg->iface.get_device_گنتی(reg);
    apir_encode_int32_t(enc, &dev_گنتی);

    واپس 0;
}

uint32_t backend_device_get_گنتی(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    int32_t dev_گنتی = reg->iface.get_device_گنتی(reg);
    apir_encode_int32_t(enc, &dev_گنتی);

    واپس 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    ثابت حرف * string = dev->iface.get_name(dev);

    ثابت size_t string_سائز = لڑی_لمبائی(string) + 1;
    apir_encode_array_سائز(enc, string_سائز);
    apir_encode_char_array(enc, string, string_سائز);

    واپس 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    ثابت حرف * string = dev->iface.get_description(dev);

    ثابت size_t string_سائز = لڑی_لمبائی(string) + 1;
    apir_encode_array_سائز(enc, string_سائز);
    apir_encode_char_array(enc, string, string_سائز);

    واپس 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    واپس 0;
}

uint32_t backend_device_get_حافظہ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    size_t آزاد, کل;
    dev->iface.get_حافظہ(dev, &آزاد, &کل);

    apir_encode_سائز_t(enc, &آزاد);
    apir_encode_سائز_t(enc, &کل);

    واپس 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);

    ثابت ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    منطقی supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    واپس 0;
}

uint32_t backend_device_get_بفر_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    ggml_backend_بفر_type_t bufft = dev->iface.get_بفر_type(dev);

    apir_encode_ggml_بفر_type(enc, bufft);

    واپس 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_بفر);
    apir_encode_bool_t(enc, &props.caps.بفر_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    واپس 0;
}

uint32_t backend_device_بفر_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    خالی * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    اگر (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        واپس 1;
    }

    size_t size;
    apir_decode_سائز_t(dec, &size);
    size_t max_tensor_سائز;
    apir_decode_سائز_t(dec, &max_tensor_سائز);

    ggml_backend_بفر_t بفر;
    بفر = dev->iface.بفر_from_host_ptr(dev, shmem_ptr, size, max_tensor_سائز);

    apir_encode_ggml_بفر(enc, بفر);
    apir_encode_ggml_بفر_type(enc, بفر->buft);

    اگر (بفر) {
        apir_track_backend_بفر(بفر);
    }

    واپس 0;
}
