#अन्तर्भाव "backend-dispatched.h"
#अन्तर्भाव "backend-virgl-apir.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdint>

uint32_t backend_बफर_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(dec);

    ध्रुव वर्ण * string = buft->iface.get_name(buft);

    ध्रुव size_t string_आकार = वाक्यदैर्घ्यम्(string) + 1;
    apir_encode_array_आकार(enc, string_आकार);
    apir_encode_char_array(enc, string, string_आकार);

    प्रत्यागम 0;
}

uint32_t backend_बफर_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_आकार_t(enc, &value);

    प्रत्यागम 0;
}

uint32_t backend_बफर_type_get_max_आकार(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(dec);

    size_t value = SIZE_MAX;
    यदि (buft->iface.get_max_आकार) {
        value = buft->iface.get_max_आकार(buft);
    }

    apir_encode_आकार_t(enc, &value);

    प्रत्यागम 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_बफर_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(dec);
    ध्रुव तार्किक is_host = असत्य;

    apir_encode_bool_t(enc, &is_host);

    प्रत्यागम 0;
}

uint32_t backend_बफर_type_alloc_बफर(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(dec);

    size_t size;
    apir_decode_आकार_t(dec, &size);

    ggml_backend_बफर_t बफर;

    बफर = buft->iface.alloc_बफर(buft, size);

    apir_encode_ggml_बफर(enc, बफर);

    यदि (बफर) {
        apir_track_backend_बफर(बफर);
    }

    प्रत्यागम 0;
}

uint32_t backend_बफर_type_get_alloc_आकार(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_अप्रयुक्तम्(ctx);
    ggml_backend_बफर_type_t buft;
    buft = apir_decode_ggml_बफर_type(dec);

    ध्रुव ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    यदि (op == शून्यसूचकः) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_समुच्चयः_fatal(dec);
        प्रत्यागम 1;
    }

    size_t value;
    यदि (buft->iface.get_alloc_आकार) {
        value = buft->iface.get_alloc_आकार(buft, op);
    } अन्यथा {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_आकार_t(enc, &value);

    प्रत्यागम 0;
}
