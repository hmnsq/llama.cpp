#شامل "backend-dispatched.h"
#شامل "backend-virgl-apir.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"

#شامل <cstdint>

uint32_t backend_بفر_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    ثابت حرف * string = buft->iface.get_name(buft);

    ثابت size_t string_سائز = لڑی_لمبائی(string) + 1;
    apir_encode_array_سائز(enc, string_سائز);
    apir_encode_char_array(enc, string, string_سائز);

    واپس 0;
}

uint32_t backend_بفر_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_سائز_t(enc, &value);

    واپس 0;
}

uint32_t backend_بفر_type_get_max_سائز(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    size_t value = SIZE_MAX;
    اگر (buft->iface.get_max_سائز) {
        value = buft->iface.get_max_سائز(buft);
    }

    apir_encode_سائز_t(enc, &value);

    واپس 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_بفر_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    GGML_غیر_مستعمل(dec);
    ثابت منطقی is_host = جھوٹ;

    apir_encode_bool_t(enc, &is_host);

    واپس 0;
}

uint32_t backend_بفر_type_alloc_بفر(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    size_t size;
    apir_decode_سائز_t(dec, &size);

    ggml_backend_بفر_t بفر;

    بفر = buft->iface.alloc_بفر(buft, size);

    apir_encode_ggml_بفر(enc, بفر);

    اگر (بفر) {
        apir_track_backend_بفر(بفر);
    }

    واپس 0;
}

uint32_t backend_بفر_type_get_alloc_سائز(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_غیر_مستعمل(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    ثابت ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    اگر (op == خالی_اشاریہ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        واپس 1;
    }

    size_t value;
    اگر (buft->iface.get_alloc_سائز) {
        value = buft->iface.get_alloc_سائز(buft, op);
    } ورنہ {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_سائز_t(enc, &value);

    واپس 0;
}
