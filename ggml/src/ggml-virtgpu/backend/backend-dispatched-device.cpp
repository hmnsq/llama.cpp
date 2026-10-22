#अन्तर्भाव "backend-dispatched.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdint>

uint32_t backend_device_get_device_गणनम्(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    int32_t dev_गणनम् = reg->iface.get_device_गणनम्(reg);
    apir_encode_int32_t(enc, &dev_गणनम्);

    प्रत्यागम 0;
}

uint32_t backend_device_get_गणनम्(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    int32_t dev_गणनम् = reg->iface.get_device_गणनम्(reg);
    apir_encode_int32_t(enc, &dev_गणनम्);

    प्रत्यागम 0;
}

uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    ध्रुव वर्ण * string = dev->iface.get_name(dev);

    ध्रुव size_t string_आकार = वाक्यदैर्घ्यम्(string) + 1;
    apir_encode_array_आकार(enc, string_आकार);
    apir_encode_char_array(enc, string, string_आकार);

    प्रत्यागम 0;
}

uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    ध्रुव वर्ण * string = dev->iface.get_description(dev);

    ध्रुव size_t string_आकार = वाक्यदैर्घ्यम्(string) + 1;
    apir_encode_array_आकार(enc, string_आकार);
    apir_encode_char_array(enc, string, string_आकार);

    प्रत्यागम 0;
}

uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    uint32_t type = dev->iface.get_type(dev);
    apir_encode_uint32_t(enc, &type);

    प्रत्यागम 0;
}

uint32_t backend_device_get_स्मृतिः(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    size_t मुक्त, योग;
    dev->iface.get_स्मृतिः(dev, &मुक्त, &योग);

    apir_encode_आकार_t(enc, &मुक्त);
    apir_encode_आकार_t(enc, &योग);

    प्रत्यागम 0;
}

uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);

    ध्रुव ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    तार्किक supports_op = dev->iface.supports_op(dev, op);

    apir_encode_bool_t(enc, &supports_op);

    प्रत्यागम 0;
}

uint32_t backend_device_get_बफर_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    ggml_backend_बफर_type_t bufft = dev->iface.get_बफर_type(dev);

    apir_encode_ggml_बफर_type(enc, bufft);

    प्रत्यागम 0;
}

uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    ggml_backend_dev_props props;
    dev->iface.get_props(dev, &props);

    apir_encode_bool_t(enc, &props.caps.async);
    apir_encode_bool_t(enc, &props.caps.host_बफर);
    apir_encode_bool_t(enc, &props.caps.बफर_from_host_ptr);
    apir_encode_bool_t(enc, &props.caps.events);
    apir_encode_bool_t(enc, &props.caps.mmap_support);

    प्रत्यागम 0;
}

uint32_t backend_device_बफर_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    शून्य * shmem_ptr = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    यदि (!shmem_ptr) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_समुच्चयः_fatal(dec);
        प्रत्यागम 1;
    }

    size_t size;
    apir_decode_आकार_t(dec, &size);
    size_t max_tensor_आकार;
    apir_decode_आकार_t(dec, &max_tensor_आकार);

    ggml_backend_बफर_t बफर;
    बफर = dev->iface.बफर_from_host_ptr(dev, shmem_ptr, size, max_tensor_आकार);

    apir_encode_ggml_बफर(enc, बफर);
    apir_encode_ggml_बफर_type(enc, बफर->buft);

    यदि (बफर) {
        apir_track_backend_बफर(बफर);
    }

    प्रत्यागम 0;
}
