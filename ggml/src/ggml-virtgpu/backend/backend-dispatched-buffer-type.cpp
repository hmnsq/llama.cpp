#ضمن "backend-dispatched.h"
#ضمن "backend-virgl-apir.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"

#ضمن <cstdint>

uint32_t backend_بفر_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_بن_ورتیا(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    ثابت حرف * string = buft->iface.get_name(buft);

    ثابت size_t string_ناپ = لڑی_لمبائی(string) + 1;
    apir_encode_array_ناپ(enc, string_ناپ);
    apir_encode_char_array(enc, string, string_ناپ);

    أرجع 0;
}

uint32_t backend_بفر_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_بن_ورتیا(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    size_t value = buft->iface.get_alignment(buft);
    apir_encode_ناپ_t(enc, &value);

    أرجع 0;
}

uint32_t backend_بفر_type_get_max_ناپ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_بن_ورتیا(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    size_t value = SIZE_MAX;
    إذا (buft->iface.get_max_ناپ) {
        value = buft->iface.get_max_ناپ(buft);
    }

    apir_encode_ناپ_t(enc, &value);

    أرجع 0;
}

/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_بفر_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_بن_ورتیا(ctx);
    GGML_بن_ورتیا(dec);
    ثابت منطقي is_host = خطأ;

    apir_encode_bool_t(enc, &is_host);

    أرجع 0;
}

uint32_t backend_بفر_type_alloc_بفر(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_بن_ورتیا(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    size_t size;
    apir_decode_ناپ_t(dec, &size);

    ggml_backend_بفر_t بفر;

    بفر = buft->iface.alloc_بفر(buft, size);

    apir_encode_ggml_بفر(enc, بفر);

    إذا (بفر) {
        apir_track_backend_بفر(بفر);
    }

    أرجع 0;
}

uint32_t backend_بفر_type_get_alloc_ناپ(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_بن_ورتیا(ctx);
    ggml_backend_بفر_type_t buft;
    buft = apir_decode_ggml_بفر_type(dec);

    ثابت ggml_tensor * op = apir_decode_ggml_tensor_inplace(dec);

    // Check for decode error
    إذا (op == خالی_سوچک) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to decode tensor\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        أرجع 1;
    }

    size_t value;
    إذا (buft->iface.get_alloc_ناپ) {
        value = buft->iface.get_alloc_ناپ(buft, op);
    } وإلا {
        value = ggml_nbytes(op);  // Default fallback
    }

    apir_encode_ناپ_t(enc, &value);

    أرجع 0;
}
